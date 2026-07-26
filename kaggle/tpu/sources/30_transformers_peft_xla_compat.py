"""Exercise a tiny Transformers/PEFT checkpoint round-trip on a real XLA TPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_TORCH_RELEASE = "2.8.0"
EXPECTED_TORCH_XLA_RELEASE = "2.8.0"
EXPECTED_TRANSFORMERS_RELEASE = "4.51.3"
EXPECTED_PEFT_RELEASE = "0.15.2"
EXPECTED_ACCELERATE_RELEASE = "1.6.0"
EXPECTED_SAFETENSORS_RELEASE = "0.5.3"
EXPECTED_TOKENIZERS_RELEASE = "0.21.4"
EXPECTED_TPU_DEVICE_COUNT = 8
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def release_version(version: str) -> str:
    """Return the public release component from a package version."""

    release = version.split("+", maxsplit=1)[0].strip()
    if not release:
        raise ValueError("package version must not be empty")
    return release


def _require_exact_release(name: str, observed: str, expected: str) -> None:
    observed_release = release_version(observed)
    if observed_release != expected:
        raise RuntimeError(f"{name} release mismatch: expected={expected!r}, observed={observed!r}")


def _sha256_named_tensors(named_tensors: list[tuple[str, Any]], torch: Any) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        payload = value.view(torch.uint8).numpy().tobytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(json.dumps(list(value.shape)).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(payload)
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _checkpoint_manifest(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _memory_info(xm: Any, device: Any) -> dict[str, int] | None:
    try:
        return {key: int(value) for key, value in xm.get_memory_info(device).items()}
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_compatibility_gate(steps: int = 12) -> dict[str, object]:
    """Run the compatibility gate and return its compact evidence report."""

    if steps < 2:
        raise ValueError("steps must be at least 2")
    if not os.environ.get("UV_RUN_RECURSION_DEPTH"):
        raise RuntimeError("TPU compatibility gate must be invoked through uv run")

    source_commit = os.environ.get("SLAO_TPU_SOURCE_COMMIT", "").strip().lower()
    if SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise RuntimeError("SLAO_TPU_SOURCE_COMMIT must be an exact 40-hex Git commit")

    import accelerate
    import peft
    import safetensors
    import tokenizers
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import transformers
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, LlamaConfig

    versions = {
        "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", ""),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "safetensors": safetensors.__version__,
        "tokenizers": tokenizers.__version__,
    }
    expected_versions = {
        "torch": EXPECTED_TORCH_RELEASE,
        "torch_xla": EXPECTED_TORCH_XLA_RELEASE,
        "transformers": EXPECTED_TRANSFORMERS_RELEASE,
        "peft": EXPECTED_PEFT_RELEASE,
        "accelerate": EXPECTED_ACCELERATE_RELEASE,
        "safetensors": EXPECTED_SAFETENSORS_RELEASE,
        "tokenizers": EXPECTED_TOKENIZERS_RELEASE,
    }
    for name, expected in expected_versions.items():
        _require_exact_release(name, versions[name], expected)

    device_type = str(xr.device_type()).upper()
    if device_type != "TPU":
        raise RuntimeError(f"expected TPU runtime, observed {device_type!r}")

    device = torch_xla.device()
    supported_devices = [str(item) for item in xm.get_xla_supported_devices()]
    device_counts = {
        "addressable": int(xr.addressable_device_count()),
        "global": int(xr.global_device_count()),
        "global_runtime": int(xr.global_runtime_device_count()),
        "supported": len(supported_devices),
    }
    visible_device_count = max(device_counts.values())
    if visible_device_count != EXPECTED_TPU_DEVICE_COUNT:
        raise RuntimeError(
            "TpuV5E8 topology mismatch: "
            f"expected={EXPECTED_TPU_DEVICE_COUNT}, observed_counts={device_counts}"
        )

    seed = 20260726
    torch.manual_seed(seed)
    torch_xla.manual_seed(seed, device=device)
    started = time.perf_counter()
    memory_before = _memory_info(xm, device)

    model_config = {
        "model_type": "llama",
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "rank": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "dtype": "torch.bfloat16",
        "attention_implementation": "eager",
        "batch_size": 2,
        "sequence_length": 8,
        "steps": steps,
    }

    with tempfile.TemporaryDirectory(prefix="slao-tpu-transformers-peft-") as temporary:
        checkpoint_root = Path(temporary)
        base_checkpoint = checkpoint_root / "base"
        adapter_checkpoint = checkpoint_root / "adapter"

        config = LlamaConfig(
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            intermediate_size=model_config["intermediate_size"],
            num_hidden_layers=model_config["num_hidden_layers"],
            num_attention_heads=model_config["num_attention_heads"],
            num_key_value_heads=model_config["num_key_value_heads"],
            max_position_embeddings=model_config["max_position_embeddings"],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
            use_cache=False,
        )
        base_model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=model_config["attention_implementation"],
        )
        base_model.save_pretrained(base_checkpoint, safe_serialization=True)
        base_model = base_model.to(device=device, dtype=torch.bfloat16)
        model = get_peft_model(
            base_model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_config["rank"],
                lora_alpha=model_config["lora_alpha"],
                lora_dropout=0.0,
                target_modules=model_config["target_modules"],
                bias="none",
            ),
        )
        model.config.use_cache = False

        named_parameters = list(model.named_parameters())
        adapter_parameters = [(name, value) for name, value in named_parameters if "lora_" in name]
        base_parameters = [(name, value) for name, value in named_parameters if "lora_" not in name]
        trainable_names = sorted(name for name, value in named_parameters if value.requires_grad)
        unexpected_trainable = [
            name for name in trainable_names if ".lora_A." not in name and ".lora_B." not in name
        ]
        if not adapter_parameters or unexpected_trainable:
            raise RuntimeError(
                "PEFT trainable-parameter invariant failed: "
                f"adapter_count={len(adapter_parameters)}, unexpected={unexpected_trainable}"
            )
        if any(value.requires_grad for _, value in base_parameters):
            raise RuntimeError("base model exposes trainable non-LoRA parameters")

        lora_a = [(name, value) for name, value in adapter_parameters if ".lora_A." in name]
        lora_b = [(name, value) for name, value in adapter_parameters if ".lora_B." in name]
        if not lora_a or not lora_b:
            raise RuntimeError("PEFT model does not expose both LoRA A and B factors")
        if not all(int(torch.count_nonzero(value.detach().cpu())) > 0 for _, value in lora_a):
            raise RuntimeError("expected every LoRA A factor to have a nonzero initialization")
        if not all(int(torch.count_nonzero(value.detach().cpu())) == 0 for _, value in lora_b):
            raise RuntimeError("expected every LoRA B factor to start at zero")

        base_hash_before = _sha256_named_tensors(base_parameters, torch)
        adapter_hash_before = _sha256_named_tensors(adapter_parameters, torch)
        inputs_cpu = torch.tensor(
            [
                [1, 11, 12, 13, 14, 15, 16, 2],
                [1, 21, 22, 23, 24, 25, 26, 2],
            ],
            dtype=torch.long,
        )
        attention_cpu = torch.ones_like(inputs_cpu)
        input_ids = inputs_cpu.to(device)
        attention_mask = attention_cpu.to(device)
        labels = inputs_cpu.to(device)
        optimizer = torch.optim.AdamW(
            [value for _, value in adapter_parameters],
            lr=0.02,
            weight_decay=0.0,
        )

        model.train()
        initial_forward_started = time.perf_counter()
        initial_loss_tensor = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        torch_xla.sync(wait=True)
        initial_loss = float(initial_loss_tensor.detach().cpu())
        initial_forward_s = time.perf_counter() - initial_forward_started

        step_losses: list[float] = []
        step_durations: list[float] = []
        for _ in range(steps):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for _, value in adapter_parameters],
                max_norm=1.0,
            )
            xm.optimizer_step(optimizer, barrier=True)
            torch_xla.sync(wait=True)
            step_losses.append(float(loss.detach().cpu()))
            step_durations.append(time.perf_counter() - step_started)

        model.eval()
        with torch.no_grad():
            final_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        torch_xla.sync(wait=True)
        final_loss = float(final_output.loss.detach().cpu())
        reference_logits = final_output.logits.detach().cpu().float()
        if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
            raise RuntimeError(
                f"non-finite compatibility loss: initial={initial_loss}, final={final_loss}"
            )
        if not final_loss < initial_loss:
            raise RuntimeError(
                "tiny language-model fit did not improve: "
                f"initial={initial_loss}, final={final_loss}"
            )

        base_hash_after = _sha256_named_tensors(base_parameters, torch)
        adapter_hash_after = _sha256_named_tensors(adapter_parameters, torch)
        if base_hash_after != base_hash_before:
            raise RuntimeError("frozen Transformers base parameters changed during LoRA training")
        if adapter_hash_after == adapter_hash_before:
            raise RuntimeError("PEFT LoRA parameters did not change during training")

        save_started = time.perf_counter()
        cpu_state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        model.save_pretrained(
            adapter_checkpoint,
            safe_serialization=True,
            state_dict=cpu_state_dict,
        )
        checkpoint_save_s = time.perf_counter() - save_started
        adapter_weight_path = adapter_checkpoint / "adapter_model.safetensors"
        if not adapter_weight_path.is_file():
            raise RuntimeError("PEFT checkpoint did not use adapter_model.safetensors")

        load_started = time.perf_counter()
        reloaded_base = AutoModelForCausalLM.from_pretrained(
            base_checkpoint,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_config["attention_implementation"],
        )
        reloaded_model = PeftModel.from_pretrained(
            reloaded_base,
            adapter_checkpoint,
            is_trainable=False,
            local_files_only=True,
        ).to(device)
        reloaded_model.config.use_cache = False
        reloaded_model.eval()
        with torch.no_grad():
            reloaded_logits_xla = reloaded_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        torch_xla.sync(wait=True)
        reloaded_logits = reloaded_logits_xla.detach().cpu().float()
        checkpoint_load_s = time.perf_counter() - load_started

        reloaded_named_parameters = list(reloaded_model.named_parameters())
        reloaded_adapter_parameters = [
            (name, value) for name, value in reloaded_named_parameters if "lora_" in name
        ]
        reloaded_base_parameters = [
            (name, value) for name, value in reloaded_named_parameters if "lora_" not in name
        ]
        reloaded_adapter_hash = _sha256_named_tensors(reloaded_adapter_parameters, torch)
        reloaded_base_hash = _sha256_named_tensors(reloaded_base_parameters, torch)
        logits_max_abs_diff = float(torch.max(torch.abs(reference_logits - reloaded_logits)).item())
        if reloaded_adapter_hash != adapter_hash_after:
            raise RuntimeError("LoRA checkpoint hash changed across save/load round-trip")
        if reloaded_base_hash != base_hash_after:
            raise RuntimeError("base-model hash changed across save/load round-trip")
        if logits_max_abs_diff > 1e-3:
            raise RuntimeError(
                f"checkpoint round-trip logits mismatch: max_abs_diff={logits_max_abs_diff}"
            )

        memory_after = _memory_info(xm, device)
        checkpoint_manifest = _checkpoint_manifest(checkpoint_root)
        report: dict[str, object] = {
            "status": "passed",
            "evidence_class": "development_transformers_peft_xla_compatibility",
            "scientific_result": False,
            "paper_comparable": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": source_commit,
            "runtime": {
                "device": str(device),
                "device_type": device_type,
                "requested_shape": "TpuV5E8",
                "expected_device_count": EXPECTED_TPU_DEVICE_COUNT,
                "visible_device_count": visible_device_count,
                "device_counts": device_counts,
                "supported_devices": supported_devices,
                "dtype": str(torch.bfloat16),
                "python": platform.python_version(),
                "uv_run_recursion_depth": os.environ["UV_RUN_RECURSION_DEPTH"],
                "versions": versions,
            },
            "model": model_config,
            "parameter_invariants": {
                "trainable_names": trainable_names,
                "trainable_parameter_count": sum(value.numel() for _, value in adapter_parameters),
                "frozen_parameter_count": sum(value.numel() for _, value in base_parameters),
                "base_sha256_before": base_hash_before,
                "base_sha256_after": base_hash_after,
                "frozen_base_preserved": True,
                "adapter_sha256_before": adapter_hash_before,
                "adapter_sha256_after": adapter_hash_after,
                "adapter_updated": True,
                "lora_a_nonzero_at_initialization": True,
                "lora_b_zero_at_initialization": True,
            },
            "training": {
                "initial_loss": initial_loss,
                "step_losses": step_losses,
                "final_loss": final_loss,
                "loss_ratio": final_loss / initial_loss,
                "finite_losses": True,
                "loss_improved": True,
            },
            "checkpoint_roundtrip": {
                "format": "Transformers base plus PEFT adapter safetensors",
                "temporary_artifacts_cleaned_after_gate": True,
                "files": checkpoint_manifest,
                "adapter_sha256_before_save": adapter_hash_after,
                "adapter_sha256_after_reload": reloaded_adapter_hash,
                "base_sha256_before_save": base_hash_after,
                "base_sha256_after_reload": reloaded_base_hash,
                "logits_max_abs_diff": logits_max_abs_diff,
                "logits_atol": 1e-3,
                "passed": True,
            },
            "timing": {
                "initial_forward_compile_s": initial_forward_s,
                "first_optimizer_step_s": step_durations[0],
                "median_post_first_optimizer_step_s": statistics.median(step_durations[1:]),
                "checkpoint_save_s": checkpoint_save_s,
                "checkpoint_load_and_inference_s": checkpoint_load_s,
                "total_elapsed_s": time.perf_counter() - started,
            },
            "memory": {
                "before": memory_before,
                "after": memory_after,
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "SLAO_TPU_COMPAT_OUTPUT",
                "/kaggle/working/tpu_transformers_peft_compat.json",
            )
        ),
    )
    args = parser.parse_args()
    report = run_compatibility_gate(steps=args.steps)
    _write_json_atomic(args.output, report)
    print("SLAO_TPU_TRANSFORMERS_PEFT_COMPAT " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
