"""Single-GPU SLAO/SeqLoRA reproduction entrypoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import time
import traceback
from contextlib import nullcontext
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
from slao_repro.slao import MergedBInitSLAOMerger, SLAOMerger


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name in {"bfloat16", "float16"}:
        return torch.float16
    return torch.float32


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
    device: torch.device,
    seed: int,
    max_optimizer_steps: int | None,
) -> dict[str, float | int]:
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
    parameter_dtype = next(model.parameters()).dtype
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and parameter_dtype == torch.float16
    )
    stop = False
    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=parameter_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                loss = model(**batch).loss
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            micro_steps += 1
            loss_sum += float(loss.detach().cpu())
            is_boundary = micro_steps % accumulation == 0 or batch_index == len(loader) - 1
            if is_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(train_config["clip_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps:
                    stop = True
                    break
        if stop:
            break
    return {
        "mean_microbatch_loss": loss_sum / max(micro_steps, 1),
        "micro_steps": micro_steps,
        "optimizer_steps": optimizer_steps,
        "elapsed_s": time.monotonic() - started,
    }


@torch.inference_mode()
def _evaluate_task(
    model: torch.nn.Module,
    examples: list[SuperNIExample],
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
    classification: bool,
) -> tuple[float, list[dict[str, Any]]]:
    train_config = config["training"]
    eval_config = config["evaluation"]
    loader = DataLoader(
        examples,
        batch_size=int(train_config["eval_batch_size"]),
        shuffle=False,
        collate_fn=EvalCollator(tokenizer, int(train_config["max_source_length"])),
    )
    predictions: list[str] = []
    references: list[tuple[str, ...]] = []
    records: list[dict[str, Any]] = []
    model.eval()
    model.config.use_cache = True
    for batch in loader:
        source_examples = batch.pop("examples")
        inputs = {key: value.to(device) for key, value in batch.items()}
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=int(eval_config["num_beams"]),
            max_new_tokens=int(eval_config["generation_max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
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


def _load_resume_inputs(
    args: argparse.Namespace,
    task_order: list[str],
    config_sha256: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float, dict[str, Any] | None]:
    if args.resume_checkpoint is None:
        if args.resume_metrics is not None or args.resume_predictions is not None:
            raise ValueError("resume metrics/predictions require --resume-checkpoint")
        return None, [], 0.0, None
    if args.resume_metrics is None:
        raise ValueError("--resume-checkpoint requires --resume-metrics")

    checkpoint_path = Path(args.resume_checkpoint).resolve()
    metrics_path = Path(args.resume_metrics).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint missing: {checkpoint_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"resume metrics missing: {metrics_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint must contain a mapping")
    if payload.get("method") != args.method or int(payload.get("seed", -1)) != args.seed:
        raise ValueError("resume checkpoint method/seed does not match this run")
    task_index = int(payload.get("task_index", 0))
    if task_index < 1 or task_index > len(task_order):
        raise ValueError("resume checkpoint task_index is outside the configured task order")
    checkpoint_order = payload.get("task_order")
    if checkpoint_order is not None and list(checkpoint_order) != task_order:
        raise ValueError("resume checkpoint task order does not match this run")
    checkpoint_config_sha = payload.get("config_sha256")
    if checkpoint_config_sha is not None and checkpoint_config_sha != config_sha256:
        raise ValueError("resume checkpoint config hash does not match this run")
    if not isinstance(payload.get("fine_tuned"), dict):
        raise ValueError("resume checkpoint is missing the fine-tuned adapter state")
    score_matrix = payload.get("score_matrix")
    if not isinstance(score_matrix, list) or len(score_matrix) != task_index:
        raise ValueError("resume score matrix length does not match task_index")
    if any(not isinstance(row, list) or len(row) != len(task_order) for row in score_matrix):
        raise ValueError("resume score matrix has an incompatible shape")
    merging_methods = {"slao", "slao_merged_b_init", "ftba_mb"}
    if args.method in merging_methods and not isinstance(payload.get("merger"), dict):
        raise ValueError("resume checkpoint is missing merger state")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"resume metrics line {line_number} is not a mapping")
        records.append(record)
    if len(records) != task_index:
        raise ValueError("resume metrics row count does not match checkpoint task_index")
    for expected_index, record in enumerate(records, start=1):
        if int(record.get("task_index", -1)) != expected_index:
            raise ValueError("resume metrics task indices are not contiguous")
        if record.get("task") != task_order[expected_index - 1]:
            raise ValueError("resume metrics task order does not match this run")
    elapsed_s = float(records[-1]["elapsed_s"])
    if elapsed_s < 0:
        raise ValueError("resume elapsed time must be non-negative")

    predictions_path = None
    predictions_sha256 = None
    if args.resume_predictions is not None:
        predictions_path = Path(args.resume_predictions).resolve()
        if not predictions_path.is_file():
            raise FileNotFoundError(f"resume predictions missing: {predictions_path}")
        predictions_sha256 = _sha256(predictions_path)

    provenance = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "predictions_path": str(predictions_path) if predictions_path is not None else None,
        "predictions_sha256": predictions_sha256,
        "completed_tasks": task_index,
        "prior_elapsed_s": elapsed_s,
    }
    return payload, records, elapsed_s, provenance


def run(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    config_sha256 = _sha256(config_path)
    if config["data"]["benchmark"] != "superni":
        raise ValueError("the current paper runner supports the pinned SuperNI benchmark")
    _seed_everything(args.seed)
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
    resume_payload, resume_records, prior_elapsed_s, resume_provenance = _load_resume_inputs(
        args, task_order, config_sha256
    )

    output_dir = Path(args.output_dir or f"outputs/{args.run_label}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_nonempty_output:
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["data"]["root"]).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"benchmark data missing at {data_root}; run fetch_benchmark_data.py first"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(str(config["model"]["dtype"]), device)
    hardware = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        ),
        "platform": platform.platform(),
    }
    manifest = {
        "run_label": args.run_label,
        "method": args.method,
        "seed": args.seed,
        "started_at_utc": _utc_now(),
        "git_revision": _git_revision(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "config": config,
        "task_order_executed": task_order,
        "resume": resume_provenance,
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
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if bool(config["model"]["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()
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
    model.to(device)
    _emit(
        "model_ready",
        model_id=config["model"]["id"],
        revision=config["model"]["revision"],
        dtype=str(dtype),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    classification_tasks = set(config["data"]["classification_tasks"])
    eval_cache: dict[str, list[SuperNIExample]] = {}
    score_matrix: list[list[float | None]] = (
        list(resume_payload["score_matrix"]) if resume_payload is not None else []
    )
    task_records: list[dict[str, Any]] = []
    merger_class = (
        MergedBInitSLAOMerger if args.method == "slao_merged_b_init" else SLAOMerger
    )
    merger = (
        merger_class.from_state_dict(resume_payload["merger"])
        if resume_payload is not None and args.method in {"slao", "slao_merged_b_init", "ftba_mb"}
        else merger_class()
    )
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.jsonl"
    if resume_records:
        metrics_path.write_text(
            "".join(json.dumps(record) + "\n" for record in resume_records),
            encoding="utf-8",
        )
    if resume_payload is not None and args.resume_predictions is not None:
        shutil.copyfile(Path(args.resume_predictions).resolve(), predictions_path)
    if resume_payload is not None:
        set_adapter_state(model, resume_payload["fine_tuned"])
    start_position = len(score_matrix) + 1
    started = time.monotonic()

    for task_position, task in enumerate(
        task_order[start_position - 1 :], start=start_position
    ):
        task_seed = args.seed * 1000 + task_position
        if args.method in {"slao", "slao_merged_b_init"} and task_position > 1:
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
            device,
            task_seed,
            args.max_optimizer_steps_per_task,
        )
        fine_tuned = capture_adapter_state(model)
        if args.method in {"slao", "slao_merged_b_init"}:
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
                device,
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
            "eval_samples": {
                name: len(eval_cache[name]) for name in task_order[:task_position]
            },
            "train": train_stats,
            "scores_percent": {
                name: row[index] for index, name in enumerate(task_order[:task_position])
            },
            "elapsed_s": prior_elapsed_s + time.monotonic() - started,
        }
        task_records.append(task_record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(task_record) + "\n")
        _emit("task_complete", **task_record)
        checkpoint_payload: dict[str, Any] = {
            "format": "slao-repro-adapter-v2",
            "method": args.method,
            "seed": args.seed,
            "task_index": task_position,
            "task_order": task_order,
            "config_sha256": config_sha256,
            "fine_tuned": fine_tuned,
            "score_matrix": score_matrix,
        }
        if args.method in {"slao", "slao_merged_b_init", "ftba_mb"}:
            checkpoint_payload["merger"] = merger.state_dict()
        torch.save(checkpoint_payload, output_dir / "adapter_checkpoint.pt")

    final_scores = [float(value) for value in score_matrix[-1] if value is not None]
    final_aa = aa(final_scores)
    final_bwt = bwt(score_matrix) if len(score_matrix) > 1 else None
    target = config["evaluation"].get("paper_target", {})
    target_value = float(target["value_percent"]) if target else None
    full_paper_order = len(task_order) == len(config["data"]["task_order"])
    comparison = None
    if target_value is not None and full_paper_order and args.method == "slao":
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
        "evidence_class": config["experiment"]["evidence_class"],
        "method": args.method,
        "seed": args.seed,
        "tasks_completed": len(score_matrix),
        "task_order": task_order,
        "aa_percent": final_aa,
        "bwt_percentage_points": final_bwt,
        "score_matrix_percent": score_matrix,
        "paper_comparison": comparison,
        "runtime_s": prior_elapsed_s + time.monotonic() - started,
        "finished_at_utc": _utc_now(),
        "hardware": hardware,
        "result_is_paper_cell": full_paper_order,
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
    parser.add_argument(
        "--method",
        choices=["slao", "slao_merged_b_init", "seqlora", "ftba_mb"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-optimizer-steps-per-task", type=int)
    parser.add_argument("--allow-nonempty-output", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-metrics")
    parser.add_argument("--resume-predictions")
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
