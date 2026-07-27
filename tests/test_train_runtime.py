from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slao_repro import train as train_module
from slao_repro.data import SuperNIExample
from slao_repro.runtime import AcceleratorRuntime
from slao_repro.train import _resolve_dtype, _train_one_task


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


def test_xla_preserves_bfloat16() -> None:
    assert _resolve_dtype("bfloat16", torch.device("xla")) is torch.bfloat16


def test_xla_rejects_float16() -> None:
    with pytest.raises(ValueError, match="requires bfloat16 or float32"):
        _resolve_dtype("float16", torch.device("xla"))


def test_runner_routes_gradient_checkpointing_through_runtime() -> None:
    source = inspect.getsource(train_module.run)

    assert "runtime.enable_gradient_checkpointing(model)" in source
    assert "model.gradient_checkpointing_enable()" not in source


def test_cpu_training_loop_uses_runtime_step_and_emits_timing(monkeypatch) -> None:
    class TinyTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, text, **_kwargs):
            return {"input_ids": [3 + (ord(character) % 8) for character in text[:3]]}

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(16, 4)
            self.output = nn.Linear(4, 16)
            self.config = SimpleNamespace(use_cache=True)

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask
            logits = self.output(self.embedding(input_ids))
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            return SimpleNamespace(loss=loss)

    model = TinyModel()
    parameters = list(model.parameters())
    monkeypatch.setattr(train_module, "adapter_parameters", lambda _model: parameters)
    runtime = AcceleratorRuntime(
        requested="cpu",
        kind="cpu",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    examples = [
        SuperNIExample(
            task="toy",
            prompt=f"p{index}",
            references=("a",),
            target="a",
            example_id=str(index),
        )
        for index in range(4)
    ]
    config = {
        "training": {
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "train_batch_size": 1,
            "max_source_length": 8,
            "max_target_length": 4,
            "gradient_accumulation_steps": 2,
            "epochs": 1,
            "clip_grad_norm": 1.0,
        }
    }

    stats = _train_one_task(
        model,
        examples,
        TinyTokenizer(),
        config,
        runtime,
        seed=7,
        max_optimizer_steps=None,
    )

    assert stats["micro_steps"] == 4
    assert stats["optimizer_steps"] == 2
    assert stats["mean_microbatch_loss"] > 0
    assert stats["first_optimizer_step_s"] > 0
    assert stats["median_post_first_optimizer_step_s"] > 0
    assert stats["runtime_memory_before"] is None
    assert stats["runtime_memory_after"] is None
