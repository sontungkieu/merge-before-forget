from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from slao_repro import runtime as runtime_module
from slao_repro.runtime import AcceleratorRuntime, resolve_runtime


def test_auto_runtime_preserves_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    runtime = resolve_runtime("auto", "bfloat16")

    assert runtime.kind == "cpu"
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype is torch.float32


def test_explicit_cuda_runtime_fails_closed_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA runtime requested"):
        resolve_runtime("cuda", "bfloat16")


def test_xla_runtime_uses_coupled_modules_and_reports_topology(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeTorchXLA:
        __version__ = "2.8.0"

        @staticmethod
        def device() -> torch.device:
            return torch.device("xla:0")

        @staticmethod
        def manual_seed(seed: int, *, device: torch.device) -> None:
            calls["seed"] = (seed, str(device))

        @staticmethod
        def sync(*, wait: bool) -> None:
            calls["sync_wait"] = wait

    class FakeXM:
        @staticmethod
        def get_xla_supported_devices() -> list[str]:
            return [f"TPU:{index}" for index in range(8)]

        @staticmethod
        def get_memory_info(device: torch.device) -> dict[str, int]:
            calls["memory_device"] = str(device)
            return {"bytes_used": 11, "bytes_limit": 22}

        @staticmethod
        def optimizer_step(optimizer: torch.optim.Optimizer, *, barrier: bool) -> None:
            calls["optimizer_barrier"] = barrier
            optimizer.step()

    fake_xr = SimpleNamespace(
        device_type=lambda: "TPU",
        addressable_device_count=lambda: 8,
        global_device_count=lambda: 8,
        global_runtime_device_count=lambda: 8,
    )
    monkeypatch.setattr(
        runtime_module,
        "_load_xla_modules",
        lambda: (FakeTorchXLA, FakeXM, fake_xr),
    )

    runtime = resolve_runtime("xla", "bfloat16")
    runtime.manual_seed(42)
    runtime.sync(wait=True)
    info = runtime.describe("bfloat16")

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor(2.0)
    runtime.optimizer_step(optimizer, runtime.grad_scaler())

    assert runtime.kind == "xla"
    assert runtime.device == torch.device("xla:0")
    assert runtime.dtype is torch.bfloat16
    assert runtime.memory_info() == {"bytes_used": 11, "bytes_limit": 22}
    assert calls == {
        "seed": (42, "xla:0"),
        "sync_wait": True,
        "optimizer_barrier": True,
        "memory_device": "xla:0",
    }
    assert parameter.item() == pytest.approx(0.8)
    assert info["torch_xla_version"] == "2.8.0"
    assert info["xla_device_type"] == "TPU"
    assert info["xla_visible_device_count"] == 8
    assert info["selected_device_count"] == 1
    assert info["uses_all_visible_devices"] is False


def test_xla_runtime_rejects_non_tpu_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_load_xla_modules",
        lambda: (
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(device_type=lambda: "CPU"),
        ),
    )

    with pytest.raises(RuntimeError, match="observed XLA device type"):
        resolve_runtime("xla", "bfloat16")


def test_xla_runtime_uses_xla_native_gradient_checkpointing(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def xla_checkpoint(function, *args, **kwargs):
        return function(*args, **kwargs)

    class FakeModel:
        def gradient_checkpointing_enable(self) -> None:
            calls["enabled"] = True

        def _set_gradient_checkpointing(self, **kwargs) -> None:
            calls["setter"] = kwargs

    monkeypatch.setattr(runtime_module, "_load_xla_checkpoint", lambda: xla_checkpoint)
    runtime = AcceleratorRuntime(
        requested="xla",
        kind="xla",
        device=torch.device("xla:0"),
        dtype=torch.bfloat16,
    )

    runtime.enable_gradient_checkpointing(FakeModel())

    assert calls == {
        "enabled": True,
        "setter": {
            "enable": True,
            "gradient_checkpointing_func": xla_checkpoint,
        },
    }


def test_cpu_runtime_keeps_transformers_gradient_checkpointing() -> None:
    calls: list[str] = []

    class FakeModel:
        def gradient_checkpointing_enable(self) -> None:
            calls.append("enabled")

    runtime = AcceleratorRuntime(
        requested="cpu",
        kind="cpu",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    runtime.enable_gradient_checkpointing(FakeModel())

    assert calls == ["enabled"]


def test_evaluation_context_preserves_inference_mode_off_xla() -> None:
    runtime = AcceleratorRuntime(
        requested="cpu",
        kind="cpu",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with runtime.evaluation_context():
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()


def test_xla_evaluation_context_avoids_inference_tensors() -> None:
    runtime = AcceleratorRuntime(
        requested="xla",
        kind="xla",
        device=torch.device("xla:0"),
        dtype=torch.bfloat16,
    )

    with runtime.evaluation_context():
        assert not torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()


def test_checkpoint_is_atomic_and_cpu_portable(tmp_path) -> None:
    runtime = AcceleratorRuntime(
        requested="cpu",
        kind="cpu",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    path = tmp_path / "adapter_checkpoint.pt"

    runtime.save_checkpoint(
        {
            "format": "test",
            "nested": {"tensor": torch.arange(4, dtype=torch.bfloat16)},
        },
        path,
    )
    restored = torch.load(path, map_location="cpu", weights_only=False)

    assert restored["format"] == "test"
    assert restored["nested"]["tensor"].device.type == "cpu"
    assert restored["nested"]["tensor"].dtype is torch.bfloat16
    assert not path.with_suffix(".pt.tmp").exists()
