"""SLAO/SeqLoRA reproduction entrypoint for CPU, CUDA, and PyTorch/XLA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from slao_repro.data import SuperNIExample, load_superni_task
from slao_repro.metrics import aa, bwt, symmetric_relative_delta, task_metric
from slao_repro.modeling import adapter_parameters, capture_adapter_state, set_adapter_state
from slao_repro.runtime import AcceleratorRuntime, resolve_dtype, resolve_runtime
from slao_repro.slao import SLAOMerger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _emit(event: str, **fields: Any) -> None:
    print("SLAO_EVENT " + json.dumps({"event": event, "at_utc": _utc_now(), **fields}), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _seed_everything(seed: int, runtime: AcceleratorRuntime | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if runtime is not None:
        runtime.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Backward-compatible import surface for the runtime dtype gate."""

    return resolve_dtype(name, device)


class TrainCollator:
    def __init__(self, tokenizer: Any, max_source: int, max_target: int) -> None:
        self.tokenizer = tokenizer
        self.max_source = max_source
        self.max_target = max_target

    def __call__(self, examples: list[SuperNIExample]) -> dict[str, torch.Tensor]:
        rows: list[tuple[list[int], list[int]]] = []
        for example in examples:
            prompt_ids = self.tokenizer(
                example.prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_source,
            )["input_ids"]
            target_ids = self.tokenizer(
                example.target,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_target - 1,
            )["input_ids"] + [self.tokenizer.eos_token_id]
            ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            rows.append((ids, labels))
        width = max(len(ids) for ids, _ in rows)
        input_ids, labels, masks = [], [], []
        for ids, row_labels in rows:
            padding = width - len(ids)
            input_ids.append([self.tokenizer.pad_token_id] * padding + ids)
            labels.append([-100] * padding + row_labels)
            masks.append([0] * padding + [1] * len(ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class EvalCollator:
    def __init__(self, tokenizer: Any, max_source: int) -> None:
        self.tokenizer = tokenizer
        self.max_source = max_source

    def __call__(self, examples: list[SuperNIExample]) -> dict[str, Any]:
        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            [example.prompt for example in examples],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_source,
            padding=True,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = previous_side
        encoded["examples"] = examples
        return encoded


def _train_one_task(
    model: torch.nn.Module,
    examples: list[SuperNIExample],
    tokenizer: Any,
    config: dict[str, Any],
    runtime: AcceleratorRuntime,
    seed: int,
    max_optimizer_steps: int | None,
) -> dict[str, Any]:
    train_config = config["training"]
    parameters = adapter_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        examples,
        batch_size=int(train_config["train_batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=runtime.is_cuda,
        collate_fn=TrainCollator(
            tokenizer,
            int(train_config["max_source_length"]),
            int(train_config["max_target_length"]),
        ),
    )
    accumulation = int(train_config["gradient_accumulation_steps"])
    epochs = int(train_config["epochs"])
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_steps = 0
    loss_sum = 0.0
    started = time.monotonic()
    scaler = runtime.grad_scaler()
    optimizer_step_durations: list[float] = []
    optimizer_window_started: float | None = None
    window_loss: torch.Tensor | None = None
    memory_before = runtime.memory_info()
    stop = False
    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            if optimizer_window_started is None:
                optimizer_window_started = time.monotonic()
            batch = runtime.move_batch(batch)
            with runtime.autocast():
                loss = model(**batch).loss
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            micro_steps += 1
            detached_loss = loss.detach()
            window_loss = detached_loss if window_loss is None else window_loss + detached_loss
            is_boundary = micro_steps % accumulation == 0 or batch_index == len(loader) - 1
            if is_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(train_config["clip_grad_norm"]))
                runtime.optimizer_step(optimizer, scaler)
                runtime.sync(wait=True)
                assert window_loss is not None
                loss_sum += float(window_loss.cpu())
                window_loss = None
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                assert optimizer_window_started is not None
                optimizer_step_durations.append(time.monotonic() - optimizer_window_started)
                optimizer_window_started = None
                if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                    stop = True
                    break
        if stop:
            break
    runtime.sync(wait=True)
    memory_after = runtime.memory_info()
    return {
        "mean_microbatch_loss": loss_sum / max(micro_steps, 1),
        "micro_steps": micro_steps,
        "optimizer_steps": optimizer_steps,
        "elapsed_s": time.monotonic() - started,
        "first_optimizer_step_s": (
            optimizer_step_durations[0] if optimizer_step_durations else None
        ),
        "median_post_first_optimizer_step_s": (
            statistics.median(optimizer_step_durations[1:])
            if len(optimizer_step_durations) > 1
            else None
        ),
        "runtime_memory_before": memory_before,
        "runtime_memory_after": memory_after,
    }


@torch.inference_mode()
def _evaluate_task(
    model: torch.nn.Module,
    examples: list[SuperNIExample],
    tokenizer: Any,
    config: dict[str, Any],
    runtime: AcceleratorRuntime,
    classification: bool,
) -> tuple[float, list[dict[str, Any]]]:
    train_config = config["training"]
    eval_config = config["evaluation"]
    loader = DataLoader(
        examples,
        batch_size=int(train_config["eval_batch_size"]),
        shuffle=False,
        pin_memory=runtime.is_cuda,
        collate_fn=EvalCollator(tokenizer, int(train_config["max_source_length"])),
    )
    predictions: list[str] = []
    references: list[tuple[str, ...]] = []
    records: list[dict[str, Any]] = []
    model.eval()
    model.config.use_cache = True
    for batch in loader:
        source_examples = batch.pop("examples")
        inputs = runtime.move_batch(batch)
        with runtime.autocast():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=int(eval_config["num_beams"]),
                max_new_tokens=int(eval_config["generation_max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        runtime.sync(wait=True)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens.detach().cpu(), skip_special_tokens=True)
        for example, prediction in zip(source_examples, decoded, strict=True):
            clean_prediction = prediction.strip()
            predictions.append(clean_prediction)
            references.append(example.references)
            records.append(
                {
                    "task": example.task,
                    "example_id": example.example_id,
                    "prediction": clean_prediction,
                    "references": list(example.references),
                }
            )
    return task_metric(predictions, references, classification=classification), records


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    if config["data"]["benchmark"] != "superni":
        raise ValueError("the current paper runner supports the pinned SuperNI benchmark")
    requested_dtype = str(config["model"]["dtype"])
    runtime = resolve_runtime(args.runtime, requested_dtype)
    _seed_everything(args.seed, runtime)
    task_order = list(config["data"]["task_order"])
    if args.max_tasks is not None:
        task_order = task_order[: args.max_tasks]
    if not task_order:
        raise ValueError("task order is empty")
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_train_samples is not None:
        config["data"]["max_train_samples_per_task"] = args.max_train_samples
    if args.max_eval_samples is not None:
        config["data"]["max_eval_samples_per_task"] = args.max_eval_samples

    output_dir = Path(args.output_dir or f"outputs/{args.run_label}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_nonempty_output:
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["data"]["root"]).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"benchmark data missing at {data_root}; run fetch_benchmark_data.py first"
        )

    hardware = {
        **runtime.describe(requested_dtype),
        "platform": platform.platform(),
        "runtime_evidence_class": (
            "approximate_portability" if runtime.is_xla else "native_pytorch"
        ),
        "runtime_paper_comparable": not runtime.is_xla,
    }
    manifest = {
        "run_label": args.run_label,
        "method": args.method,
        "seed": args.seed,
        "started_at_utc": _utc_now(),
        "git_revision": _git_revision(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "config": config,
        "task_order_executed": task_order,
        "hardware": hardware,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _emit("run_start", run_label=args.run_label, method=args.method, hardware=hardware)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=bool(config["model"]["trust_remote_code"]),
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=runtime.dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if bool(config["model"]["gradient_checkpointing"]):
        runtime.enable_gradient_checkpointing(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias="none",
        ),
    )
    model.to(runtime.device)
    _emit(
        "model_ready",
        model_id=config["model"]["id"],
        revision=config["model"]["revision"],
        dtype=str(runtime.dtype),
        runtime_kind=runtime.kind,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    classification_tasks = set(config["data"]["classification_tasks"])
    eval_cache: dict[str, list[SuperNIExample]] = {}
    score_matrix: list[list[float | None]] = []
    task_records: list[dict[str, Any]] = []
    merger = SLAOMerger()
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()

    for task_position, task in enumerate(task_order, start=1):
        task_seed = args.seed * 1000 + task_position
        if args.method == "slao" and task_position > 1:
            set_adapter_state(model, merger.next_initial_state())
        train_examples = load_superni_task(
            data_root,
            task,
            "train",
            max_samples=int(config["data"]["max_train_samples_per_task"]),
            seed=task_seed,
        )
        _emit(
            "task_train_start",
            task_index=task_position,
            task=task,
            samples=len(train_examples),
        )
        train_stats = _train_one_task(
            model,
            train_examples,
            tokenizer,
            config,
            runtime,
            task_seed,
            args.max_optimizer_steps_per_task,
        )
        fine_tuned = capture_adapter_state(model)
        if args.method == "slao":
            eval_state = (
                merger.add_first_task(fine_tuned)
                if task_position == 1
                else merger.add_task(fine_tuned)
            )
        elif args.method == "ftba_mb":
            eval_state = (
                merger.add_first_task(fine_tuned)
                if task_position == 1
                else merger.add_task(fine_tuned)
            )
        else:
            eval_state = fine_tuned
        set_adapter_state(model, eval_state)

        row: list[float | None] = [None] * len(task_order)
        task_prediction_records: list[dict[str, Any]] = []
        for seen_index, seen_task in enumerate(task_order[:task_position]):
            if seen_task not in eval_cache:
                eval_cache[seen_task] = load_superni_task(
                    data_root,
                    seen_task,
                    "test",
                    max_samples=int(config["data"]["max_eval_samples_per_task"]),
                    seed=args.seed,
                )
            score, prediction_records = _evaluate_task(
                model,
                eval_cache[seen_task],
                tokenizer,
                config,
                runtime,
                seen_task in classification_tasks,
            )
            row[seen_index] = score
            for record in prediction_records:
                record.update(
                    {
                        "after_task_index": task_position,
                        "after_task": task,
                        "method": args.method,
                        "seed": args.seed,
                    }
                )
            task_prediction_records.extend(prediction_records)
        score_matrix.append(row)
        set_adapter_state(model, fine_tuned)
        with predictions_path.open("a", encoding="utf-8") as handle:
            for record in task_prediction_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        task_record = {
            "phase": "task_complete",
            "task_index": task_position,
            "task": task,
            "train_samples": len(train_examples),
            "eval_samples": {name: len(eval_cache[name]) for name in task_order[:task_position]},
            "train": train_stats,
            "scores_percent": {
                name: row[index] for index, name in enumerate(task_order[:task_position])
            },
            "elapsed_s": time.monotonic() - started,
        }
        task_records.append(task_record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(task_record) + "\n")
        _emit("task_complete", **task_record)
        checkpoint_payload: dict[str, Any] = {
            "format": "slao-repro-adapter-v1",
            "method": args.method,
            "seed": args.seed,
            "task_index": task_position,
            "fine_tuned": fine_tuned,
            "score_matrix": score_matrix,
        }
        if args.method in {"slao", "ftba_mb"}:
            checkpoint_payload["merger"] = merger.state_dict()
        checkpoint_payload["runtime"] = hardware
        runtime.save_checkpoint(
            checkpoint_payload,
            output_dir / "adapter_checkpoint.pt",
        )

    final_scores = [float(value) for value in score_matrix[-1] if value is not None]
    final_aa = aa(final_scores)
    final_bwt = bwt(score_matrix) if len(score_matrix) > 1 else None
    target = config["evaluation"].get("paper_target", {})
    target_value = float(target["value_percent"]) if target else None
    full_paper_order = len(task_order) == len(config["data"]["task_order"])
    comparison = None
    if (
        target_value is not None
        and full_paper_order
        and args.method == "slao"
        and not runtime.is_xla
    ):
        comparison = {
            "target_percent": target_value,
            "observed_percent": final_aa,
            "absolute_delta_percentage_points": final_aa - target_value,
            "relative_delta": symmetric_relative_delta(final_aa, target_value),
            "within_registered_stochastic_tolerance": symmetric_relative_delta(
                final_aa, target_value
            )
            < float(config["evaluation"]["stochastic_relative_tolerance"]),
        }
    summary = {
        "run_label": args.run_label,
        "status": "completed",
        "evidence_class": (
            "approximate_portability" if runtime.is_xla else config["experiment"]["evidence_class"]
        ),
        "method": args.method,
        "seed": args.seed,
        "tasks_completed": len(task_order),
        "task_order": task_order,
        "aa_percent": final_aa,
        "bwt_percentage_points": final_bwt,
        "score_matrix_percent": score_matrix,
        "paper_comparison": comparison,
        "runtime_s": time.monotonic() - started,
        "finished_at_utc": _utc_now(),
        "hardware": hardware,
        "paper_comparable": full_paper_order and not runtime.is_xla,
        "result_is_paper_cell": full_paper_order and not runtime.is_xla,
        "three_seed_protocol_complete": False,
    }
    _write_json(output_dir / "summary.json", summary)
    with (output_dir / "score_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task", *task_order])
        for task, row in zip(task_order, score_matrix, strict=True):
            writer.writerow([task, *row])
    _emit("run_complete", **summary)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["slao", "seqlora", "ftba_mb"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--runtime",
        choices=["auto", "cpu", "cuda", "xla"],
        default="auto",
        help="explicit accelerator runtime; use xla for the Kaggle TPU portability track",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-optimizer-steps-per-task", type=int)
    parser.add_argument("--allow-nonempty-output", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as error:
        output_dir = Path(args.output_dir or f"outputs/{args.run_label}").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "run_label": args.run_label,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "failure.json", failure)
        _emit("run_failed", **failure)
        raise


if __name__ == "__main__":
    main()
