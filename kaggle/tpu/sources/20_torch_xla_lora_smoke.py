"""Run a tiny, single-task LoRA overfit gate on one real XLA TPU device."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Minimal frozen linear layer with trainable low-rank A/B factors."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        if not 0 < rank <= min(in_features, out_features):
            raise ValueError("rank must be positive and no larger than either feature dimension")
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = 1.0 / rank
        nn.init.normal_(self.lora_A, mean=0.0, std=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(inputs, self.weight)
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, self.lora_A), self.lora_B
        )
        return base + self.scale * update


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def run_smoke(steps: int = 80) -> dict[str, object]:
    if not os.environ.get("UV_RUN_RECURSION_DEPTH"):
        raise RuntimeError("TPU smoke must be invoked through uv run")

    import torch_xla
    import torch_xla.runtime as xr

    if str(xr.device_type()).upper() != "TPU":
        raise RuntimeError(f"expected TPU runtime, observed {xr.device_type()!r}")

    torch.manual_seed(20260722)
    device = torch_xla.device()
    input_features = 8
    output_features = 4
    rank = 4
    sample_count = 64

    inputs_cpu = torch.randn(sample_count, input_features)
    teacher_cpu = torch.randn(output_features, input_features)
    targets_cpu = torch.nn.functional.linear(inputs_cpu, teacher_cpu)
    model = LoRALinear(input_features, output_features, rank).to(device)
    inputs = inputs_cpu.to(device)
    targets = targets_cpu.to(device)
    optimizer = torch.optim.AdamW([model.lora_A, model.lora_B], lr=0.12, weight_decay=0.0)

    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    if trainable_names != ["lora_A", "lora_B"]:
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names}")
    base_hash_before = tensor_sha256(model.weight)

    losses: list[float] = []
    step_durations: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        torch_xla.sync(wait=True)
        losses.append(float(loss.detach().cpu()))
        step_durations.append(time.perf_counter() - step_started)
    elapsed = time.perf_counter() - started

    final_loss = losses[-1]
    initial_loss = losses[0]
    loss_ratio = final_loss / initial_loss
    base_hash_after = tensor_sha256(model.weight)
    if base_hash_after != base_hash_before:
        raise RuntimeError("frozen base weight changed during LoRA smoke")
    if not final_loss < initial_loss * 0.05:
        raise RuntimeError(
            f"one-task overfit gate failed: initial={initial_loss:.8f}, final={final_loss:.8f}"
        )

    return {
        "status": "passed",
        "evidence_class": "development_tpu_lora_smoke",
        "scientific_result": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "synthetic_linear_regression_one_task_overfit",
        "device": str(device),
        "device_type": xr.device_type(),
        "torch_version": torch.__version__,
        "torch_xla_version": getattr(torch_xla, "__version__", None),
        "python": platform.python_version(),
        "uv_run_recursion_depth": os.environ["UV_RUN_RECURSION_DEPTH"],
        "sample_count": sample_count,
        "steps": steps,
        "rank": rank,
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "frozen_parameters": sum(
            value.numel() for value in model.parameters() if not value.requires_grad
        ),
        "trainable_names": trainable_names,
        "base_weight_sha256_before": base_hash_before,
        "base_weight_sha256_after": base_hash_after,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "elapsed_s": elapsed,
        "first_step_s": step_durations[0],
        "median_post_compile_step_s": sorted(step_durations[1:])[len(step_durations[1:]) // 2],
    }


def main() -> None:
    report = run_smoke()
    output = Path(os.environ.get("SLAO_TPU_SMOKE_OUTPUT", "/kaggle/working/tpu_lora_smoke.json"))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SLAO_TPU_LORA_SMOKE " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
