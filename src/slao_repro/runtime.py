"""Accelerator runtime abstraction for CPU, CUDA, and PyTorch/XLA."""

from __future__ import annotations

import importlib
import importlib.metadata
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve the configured model dtype without silently emulating TPU/CUDA modes."""

    normalized = name.strip().lower()
    if normalized not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported model dtype: {name!r}")
    if device.type == "xla":
        if normalized == "float16":
            raise ValueError("TPU/XLA runtime requires bfloat16 or float32, not float16")
        return torch.bfloat16 if normalized == "bfloat16" else torch.float32
    if device.type != "cuda":
        return torch.float32
    if normalized == "bfloat16" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    if normalized in {"bfloat16", "float16"}:
        return torch.float16
    return torch.float32


def _load_xla_modules() -> tuple[Any, Any, Any]:
    try:
        torch_xla = importlib.import_module("torch_xla")
        xm = importlib.import_module("torch_xla.core.xla_model")
        xr = importlib.import_module("torch_xla.runtime")
    except ImportError as error:
        raise RuntimeError(
            "XLA runtime requested but the coupled torch_xla package is unavailable"
        ) from error
    return torch_xla, xm, xr


def _torch_xla_version(torch_xla: Any) -> str:
    version = str(getattr(torch_xla, "__version__", "")).strip()
    if version:
        return version
    try:
        return importlib.metadata.version("torch-xla")
    except importlib.metadata.PackageNotFoundError:
        return ""


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, set):
        return {_cpu_tree(item) for item in value}
    return value


@dataclass(frozen=True)
class AcceleratorRuntime:
    """Small runtime contract used by the scientific runner."""

    requested: str
    kind: str
    device: torch.device
    dtype: torch.dtype
    torch_xla: Any | None = None
    xm: Any | None = None
    xr: Any | None = None

    @property
    def is_xla(self) -> bool:
        return self.kind == "xla"

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def manual_seed(self, seed: int) -> None:
        if self.is_xla:
            assert self.torch_xla is not None
            self.torch_xla.manual_seed(seed, device=self.device)

    def autocast(self) -> AbstractContextManager[Any]:
        if self.is_cuda and self.dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        if self.is_xla and self.dtype == torch.bfloat16:
            return torch.autocast(device_type="xla", dtype=self.dtype)
        return nullcontext()

    def grad_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            "cuda",
            enabled=self.is_cuda and self.dtype == torch.float16,
        )

    def move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        non_blocking = self.is_cuda
        return {
            key: value.to(self.device, non_blocking=non_blocking) for key, value in batch.items()
        }

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
    ) -> None:
        if self.is_xla:
            if scaler.is_enabled():
                raise RuntimeError("gradient scaling must stay disabled on TPU/XLA BF16")
            assert self.xm is not None
            self.xm.optimizer_step(optimizer, barrier=True)
        elif scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    def sync(self, *, wait: bool = True) -> None:
        if self.is_xla:
            assert self.torch_xla is not None
            self.torch_xla.sync(wait=wait)
        elif self.is_cuda:
            torch.cuda.synchronize(self.device)

    def scalar(self, value: torch.Tensor) -> float:
        self.sync(wait=True)
        return float(value.detach().cpu())

    def memory_info(self) -> dict[str, int] | None:
        if self.is_xla:
            assert self.xm is not None
            try:
                return {
                    key: int(value) for key, value in self.xm.get_memory_info(self.device).items()
                }
            except (AttributeError, RuntimeError):
                return None
        if self.is_cuda:
            properties = torch.cuda.get_device_properties(self.device)
            return {
                "bytes_used": int(torch.cuda.memory_allocated(self.device)),
                "peak_bytes_used": int(torch.cuda.max_memory_allocated(self.device)),
                "bytes_reserved": int(torch.cuda.memory_reserved(self.device)),
                "bytes_limit": int(properties.total_memory),
            }
        return None

    def save_checkpoint(self, payload: dict[str, Any], path: str | Path) -> None:
        """Synchronize and write a CPU-portable checkpoint atomically."""

        self.sync(wait=True)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(_cpu_tree(payload), temporary)
        temporary.replace(target)

    def describe(self, requested_dtype: str) -> dict[str, Any]:
        hardware: dict[str, Any] = {
            "runtime_kind": self.kind,
            "requested_runtime": self.requested,
            "device": str(self.device),
            "requested_dtype": requested_dtype,
            "resolved_dtype": str(self.dtype),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_native_bf16_supported": (
                torch.cuda.is_bf16_supported(including_emulation=False)
                if torch.cuda.is_available()
                else False
            ),
            "gpu_name": (torch.cuda.get_device_name(self.device) if self.is_cuda else None),
            "gpu_memory_bytes": (
                torch.cuda.get_device_properties(self.device).total_memory if self.is_cuda else None
            ),
            "torch_xla_version": None,
            "xla_device_type": None,
            "xla_addressable_device_count": 0,
            "xla_global_device_count": 0,
            "xla_global_runtime_device_count": 0,
            "xla_supported_devices": [],
            "selected_device_count": 1,
            "uses_all_visible_devices": True,
        }
        if self.is_xla:
            assert self.torch_xla is not None
            assert self.xm is not None
            assert self.xr is not None
            supported_devices = [str(item) for item in self.xm.get_xla_supported_devices()]
            counts = {
                "addressable": int(self.xr.addressable_device_count()),
                "global": int(self.xr.global_device_count()),
                "global_runtime": int(self.xr.global_runtime_device_count()),
                "supported": len(supported_devices),
            }
            visible_device_count = max(counts.values())
            hardware.update(
                {
                    "torch_xla_version": _torch_xla_version(self.torch_xla),
                    "xla_device_type": str(self.xr.device_type()).upper(),
                    "xla_addressable_device_count": counts["addressable"],
                    "xla_global_device_count": counts["global"],
                    "xla_global_runtime_device_count": counts["global_runtime"],
                    "xla_supported_devices": supported_devices,
                    "xla_visible_device_count": visible_device_count,
                    "uses_all_visible_devices": visible_device_count == 1,
                }
            )
        return hardware


def resolve_runtime(requested: str, requested_dtype: str) -> AcceleratorRuntime:
    """Select an explicit accelerator runtime while preserving legacy auto behavior."""

    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "xla"}:
        raise ValueError(f"unsupported runtime: {requested!r}")
    selected = normalized
    if selected == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    if selected == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
        return AcceleratorRuntime(
            requested=normalized,
            kind=selected,
            device=device,
            dtype=resolve_dtype(requested_dtype, device),
        )
    if selected == "cpu":
        device = torch.device("cpu")
        return AcceleratorRuntime(
            requested=normalized,
            kind=selected,
            device=device,
            dtype=resolve_dtype(requested_dtype, device),
        )
    torch_xla, xm, xr = _load_xla_modules()
    device_type = str(xr.device_type()).upper()
    if device_type != "TPU":
        raise RuntimeError(f"TPU/XLA runtime requested, observed XLA device type {device_type!r}")
    device = torch_xla.device()
    if getattr(device, "type", "") != "xla":
        raise RuntimeError(f"torch_xla.device() returned a non-XLA device: {device}")
    return AcceleratorRuntime(
        requested=normalized,
        kind=selected,
        device=device,
        dtype=resolve_dtype(requested_dtype, device),
        torch_xla=torch_xla,
        xm=xm,
        xr=xr,
    )
