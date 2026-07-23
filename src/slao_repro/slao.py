"""Faithful tensor-level implementation of SLAO Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TypeAlias

import torch

LayerState: TypeAlias = dict[str, torch.Tensor]
AdapterState: TypeAlias = dict[str, LayerState]


def _clone_state(state: AdapterState) -> AdapterState:
    return {
        name: {part: tensor.detach().clone() for part, tensor in parts.items()}
        for name, parts in state.items()
    }


def _validate_state(state: AdapterState) -> None:
    if not state:
        raise ValueError("adapter state must contain at least one LoRA layer")
    for name, parts in state.items():
        if set(parts) != {"A", "B"}:
            raise ValueError(f"layer {name!r} must contain exactly A and B")
        a, b = parts["A"], parts["B"]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"layer {name!r} A and B must be matrices")
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"layer {name!r} rank mismatch: A={a.shape}, B={b.shape}")


def canonical_qr_rows(previous_a: torch.Tensor) -> torch.Tensor:
    """Return Algorithm-1 QR initialization with deterministic sign normalization.

    PEFT stores A as ``[rank, in_features]``. Reduced QR is applied to ``A.T``.
    ``torch.sign(0)`` is zero, which would destroy a Q column; exact zero diagonal
    entries are therefore assigned +1. This is the only numerical convention
    added to the paper's ``sign(diag(R))`` expression.
    """

    if previous_a.ndim != 2:
        raise ValueError("A must be a rank-2 tensor")
    rank, in_features = previous_a.shape
    if rank > in_features:
        raise ValueError("reduced QR requires LoRA rank <= input dimension")
    q, r = torch.linalg.qr(previous_a.transpose(0, 1), mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    return q.transpose(0, 1).contiguous()


def time_aware_coefficient(task_index: int) -> float:
    """Paper coefficient lambda(i)=1/sqrt(i), with one-indexed task index."""

    if task_index < 1:
        raise ValueError("task_index is one-indexed and must be positive")
    return 1.0 / sqrt(task_index)


def initialize_from_last_finetuned(last_finetuned: AdapterState) -> AdapterState:
    """Initialize task i using QR(A_ft,i-1) and B_ft,i-1."""

    _validate_state(last_finetuned)
    initialized: AdapterState = {}
    for name, parts in last_finetuned.items():
        initialized[name] = {
            "A": canonical_qr_rows(parts["A"]),
            "B": parts["B"].detach().clone(),
        }
    return initialized


def merge_finetuned_state(
    previous_merged: AdapterState,
    current_finetuned: AdapterState,
    task_index: int,
) -> AdapterState:
    """Apply Algorithm-1 asymmetric merge: replace A and interpolate B."""

    _validate_state(previous_merged)
    _validate_state(current_finetuned)
    if previous_merged.keys() != current_finetuned.keys():
        raise ValueError("merged and fine-tuned states must have identical layer keys")
    coefficient = time_aware_coefficient(task_index)
    merged: AdapterState = {}
    for name in previous_merged:
        old, new = previous_merged[name], current_finetuned[name]
        if old["A"].shape != new["A"].shape or old["B"].shape != new["B"].shape:
            raise ValueError(f"shape changed for LoRA layer {name!r}")
        merged[name] = {
            "A": new["A"].detach().clone(),
            "B": old["B"] + coefficient * (new["B"] - old["B"]),
        }
    return merged


@dataclass
class SLAOMerger:
    """State machine that keeps merged and last-fine-tuned LoRAs distinct."""

    task_index: int = 0
    merged: AdapterState | None = None
    last_finetuned: AdapterState | None = None

    def add_first_task(self, fine_tuned: AdapterState) -> AdapterState:
        if self.task_index != 0:
            raise RuntimeError("first task was already registered")
        _validate_state(fine_tuned)
        self.task_index = 1
        self.merged = _clone_state(fine_tuned)
        self.last_finetuned = _clone_state(fine_tuned)
        return _clone_state(self.merged)

    def next_initial_state(self) -> AdapterState:
        if self.last_finetuned is None:
            raise RuntimeError("task 1 must be registered before requesting task 2 initialization")
        return initialize_from_last_finetuned(self.last_finetuned)

    def add_task(self, fine_tuned: AdapterState) -> AdapterState:
        if self.task_index < 1 or self.merged is None:
            raise RuntimeError("use add_first_task for task 1")
        next_index = self.task_index + 1
        self.merged = merge_finetuned_state(self.merged, fine_tuned, next_index)
        self.last_finetuned = _clone_state(fine_tuned)
        self.task_index = next_index
        return _clone_state(self.merged)

    def state_dict(self) -> dict[str, object]:
        return {
            "task_index": self.task_index,
            "merged": _clone_state(self.merged) if self.merged is not None else None,
            "last_finetuned": (
                _clone_state(self.last_finetuned) if self.last_finetuned is not None else None
            ),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> SLAOMerger:
        obj = cls(task_index=int(state["task_index"]))
        merged = state.get("merged")
        last = state.get("last_finetuned")
        obj.merged = _clone_state(merged) if isinstance(merged, dict) else None
        obj.last_finetuned = _clone_state(last) if isinstance(last, dict) else None
        if obj.task_index > 0:
            if obj.merged is None or obj.last_finetuned is None:
                raise ValueError("non-empty merger checkpoint is missing tensor state")
            _validate_state(obj.merged)
            _validate_state(obj.last_finetuned)
        return obj


@dataclass
class MergedBInitSLAOMerger(SLAOMerger):
    """Comparison variant matching the audited third-party initialization.

    Algorithm 1 initializes the next B factor from the previous fine-tuned B.
    Backpropagate at commit ``ed53fd4f82c87df6a07af34db66dfaefb1318ca6``
    instead derives both factors from its accumulated merged state.  This
    class isolates that difference while leaving the benchmark pipeline fixed.
    """

    def next_initial_state(self) -> AdapterState:
        if self.merged is None:
            raise RuntimeError("task 1 must be registered before requesting task 2 initialization")
        return initialize_from_last_finetuned(self.merged)
