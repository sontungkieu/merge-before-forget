from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slao_repro import train as train_module
from slao_repro.data import SuperNIExample
from slao_repro.runtime import AcceleratorRuntime
from slao_repro.train import TrainCollator, _resolve_dtype, _train_one_task


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


def test_evaluation_routes_grad_context_through_runtime() -> None:
    source = inspect.getsource(train_module._evaluate_task)
    implementation = inspect.getsource(train_module._evaluate_task_impl)

    assert "with runtime.evaluation_context():" in source
    assert "@torch.inference_mode()" not in source
    assert "**runtime.generation_kwargs()" in implementation


def test_xla_evaluation_offloads_to_cpu_and_restores_training_runtime(monkeypatch) -> None:
    moves: list[tuple[str, torch.dtype]] = []
    syncs: list[bool] = []
    events: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def to(self, *, device, dtype):
            moves.append((str(device), dtype))
            return self

    class FakeXLARuntime:
        is_xla = True
        kind = "xla"
        device = torch.device("xla:0")
        dtype = torch.bfloat16

        @staticmethod
        def sync(*, wait):
            syncs.append(wait)

    training_runtime = FakeXLARuntime()
    monkeypatch.setattr(
        train_module,
        "_emit",
        lambda event, **fields: events.append((event, fields)),
    )

    with train_module._evaluation_runtime(
        FakeModel(),
        training_runtime,
        "bfloat16",
    ) as evaluation_runtime:
        assert evaluation_runtime.kind == "cpu"
        assert evaluation_runtime.device == torch.device("cpu")
        assert evaluation_runtime.dtype is torch.float32

    assert moves == [
        ("cpu", torch.float32),
        ("xla:0", torch.bfloat16),
    ]
    assert syncs == [True, True]
    assert [event for event, _fields in events] == [
        "evaluation_offload_start",
        "evaluation_offload_complete",
    ]


def test_non_xla_evaluation_keeps_model_and_runtime_unchanged() -> None:
    class FakeModel:
        def to(self, **_kwargs):
            raise AssertionError("native evaluation must not move the model")

    runtime = AcceleratorRuntime(
        requested="cpu",
        kind="cpu",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    with train_module._evaluation_runtime(FakeModel(), runtime, "bfloat16") as selected:
        assert selected is runtime


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


def test_xla_training_uses_static_padding_and_bounds_accumulation_graph(monkeypatch) -> None:
    class TinyTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            ids = [3 + (ord(character) % 8) for character in text]
            return {"input_ids": ids[: kwargs.get("max_length", len(ids))]}

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

    class FakeXLARuntime:
        is_xla = True
        is_cuda = False
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.sync_calls = 0
            self.optimizer_steps = 0

        def grad_scaler(self):
            return torch.amp.GradScaler("cuda", enabled=False)

        def move_batch(self, batch):
            return batch

        def autocast(self):
            return nullcontext()

        def optimizer_step(self, optimizer, scaler):
            assert not scaler.is_enabled()
            self.optimizer_steps += 1
            optimizer.step()

        def sync(self, *, wait):
            assert wait is True
            self.sync_calls += 1

        def memory_info(self):
            return {"bytes_used": 1, "bytes_limit": 2}

    tokenizer = TinyTokenizer()
    collated = TrainCollator(
        tokenizer,
        max_source=8,
        max_target=4,
        pad_to_max_length=True,
    )(
        [
            SuperNIExample(
                task="toy",
                prompt="p",
                references=("a",),
                target="a",
                example_id="0",
            ),
            SuperNIExample(
                task="toy",
                prompt="prompt",
                references=("answer",),
                target="answer",
                example_id="1",
            ),
        ]
    )
    assert collated["input_ids"].shape == (2, 12)

    model = TinyModel()
    parameters = list(model.parameters())
    monkeypatch.setattr(train_module, "adapter_parameters", lambda _model: parameters)
    runtime = FakeXLARuntime()
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
        tokenizer,
        config,
        runtime,
        seed=7,
        max_optimizer_steps=None,
    )

    assert stats["micro_steps"] == 4
    assert stats["optimizer_steps"] == 2
    assert runtime.optimizer_steps == 2
    # Two non-boundary accumulation flushes, two optimizer-boundary syncs,
    # and the final task sync keep each XLA graph bounded.
    assert runtime.sync_calls == 5
