from __future__ import annotations

import torch

from slao_repro.train import _resolve_dtype


def test_bfloat16_requires_native_cuda_support(monkeypatch) -> None:
    calls: list[bool] = []

    def emulation_only(*, including_emulation: bool = True) -> bool:
        calls.append(including_emulation)
        return including_emulation

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", emulation_only)

    assert _resolve_dtype("bfloat16", torch.device("cuda")) is torch.float16
    assert calls == [False]


def test_bfloat16_is_kept_when_cuda_support_is_native(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda *, including_emulation=True: not including_emulation,
    )

    assert _resolve_dtype("bfloat16", torch.device("cuda")) is torch.bfloat16


def test_non_cuda_dtype_is_float32() -> None:
    assert _resolve_dtype("bfloat16", torch.device("cpu")) is torch.float32
