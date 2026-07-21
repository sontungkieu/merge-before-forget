"""Synthetic smoke, one-task overfit, and short sequential SLAO gates."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch import nn

from slao_repro.checkpoint import load_checkpoint, save_checkpoint
from slao_repro.slao import AdapterState, SLAOMerger


class ToyLoRA(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int, generator: torch.Generator):
        super().__init__()
        self.a = nn.Parameter(torch.randn(rank, input_dim, generator=generator) * 0.1)
        self.b = nn.Parameter(torch.zeros(output_dim, rank))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.a.T @ self.b.T

    def state(self) -> AdapterState:
        return {"toy": {"A": self.a.detach().clone(), "B": self.b.detach().clone()}}

    def load_state(self, state: AdapterState) -> None:
        with torch.no_grad():
            self.a.copy_(state["toy"]["A"])
            self.b.copy_(state["toy"]["B"])


@dataclass
class FitResult:
    initial_loss: float
    final_loss: float
    reduction: float


def fit(
    model: ToyLoRA,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> FitResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial = float(torch.mean((model(inputs) - targets) ** 2))
    for _ in range(steps):
        loss = torch.mean((model(inputs) - targets) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError("synthetic training produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(torch.mean((model(inputs) - targets) ** 2))
    return FitResult(initial_loss=initial, final_loss=final, reduction=1.0 - final / initial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dev/synthetic.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["training"]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    rank = int(config["rank"])
    samples = int(config["samples_per_task"])
    steps = int(config["steps_per_task"])
    learning_rate = float(config["learning_rate"])
    started = time.monotonic()

    # One-task overfit: the target is exactly representable at the configured rank.
    inputs = torch.randn(samples, input_dim, generator=generator)
    target_a = torch.randn(rank, input_dim, generator=generator)
    target_b = torch.randn(output_dim, rank, generator=generator)
    targets = inputs @ target_a.T @ target_b.T
    model = ToyLoRA(input_dim, output_dim, rank, generator)
    overfit = fit(model, inputs, targets, steps, learning_rate)
    if overfit.reduction < 0.95:
        raise RuntimeError(f"one-task overfit gate failed: reduction={overfit.reduction:.4f}")

    # Short sequential run: explicitly exercise Algorithm-1 init and merge transitions.
    merger = SLAOMerger()
    sequential = []
    for task_index in range(1, int(config["tasks"]) + 1):
        if task_index > 1:
            model.load_state(merger.next_initial_state())
        task_inputs = torch.randn(samples, input_dim, generator=generator)
        task_a = torch.randn(rank, input_dim, generator=generator)
        task_b = torch.randn(output_dim, rank, generator=generator)
        task_targets = task_inputs @ task_a.T @ task_b.T
        result = fit(model, task_inputs, task_targets, steps, learning_rate)
        fine_tuned = model.state()
        merged = (
            merger.add_first_task(fine_tuned)
            if task_index == 1
            else merger.add_task(fine_tuned)
        )
        model.load_state(merged)
        sequential.append(
            {
                "task_index": task_index,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "loss_reduction": result.reduction,
                "merged_b_norm": float(torch.linalg.vector_norm(merged["toy"]["B"])),
            }
        )

    checkpoint = output_dir / "synthetic_checkpoint.pt"
    save_checkpoint(checkpoint, merger, {"gate": "short_sequential", "tasks": len(sequential)})
    restored, metadata = load_checkpoint(checkpoint)
    if restored.task_index != merger.task_index or metadata["tasks"] != len(sequential):
        raise RuntimeError("checkpoint round-trip gate failed")
    summary = {
        "status": "completed",
        "evidence_class": "development",
        "one_task_overfit": overfit.__dict__,
        "short_sequential": sequential,
        "checkpoint_roundtrip": True,
        "runtime_s": time.monotonic() - started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SLAO_DEV_GATES " + json.dumps(summary))


if __name__ == "__main__":
    main()

