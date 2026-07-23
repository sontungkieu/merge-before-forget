# ruff: noqa: F821
"""Resume a task-boundary checkpoint attached as a Kaggle dataset."""

METHOD = "__METHOD__"
SEED = "__SEED__"
PARENT_RUN_ID = "__PARENT_RUN_ID__"
STOP_AFTER_TASK_INDEX = "__STOP_AFTER_TASK_INDEX__"

if METHOD not in {"slao", "slao_merged_b_init", "seqlora", "ftba_mb"}:
    raise ValueError(f"unsupported staged method: {METHOD}")
if not PARENT_RUN_ID:
    raise ValueError("resume run requires a parent run id")

checkpoint_candidates = list(
    Path("/kaggle/input").rglob(
        f"slao-artifacts/{PARENT_RUN_ID}/training/adapter_checkpoint.pt"
    )
)
if len(checkpoint_candidates) != 1:
    raise RuntimeError(
        f"expected one parent checkpoint for {PARENT_RUN_ID}, "
        f"found {len(checkpoint_candidates)}"
    )
checkpoint_path = checkpoint_candidates[0]
parent_training = checkpoint_path.parent
metrics_path = parent_training / "metrics.jsonl"
predictions_path = parent_training / "predictions.jsonl"
if not metrics_path.is_file():
    raise RuntimeError(f"parent metrics are missing: {metrics_path}")

run_uv(
    "python",
    "-m",
    "slao_repro.prepare_model",
    "--config",
    "configs/paper/qwen25_3b_superni_o1.yaml",
    "--output",
    ARTIFACT_ROOT / "model_prepare.json",
)
run_uv(
    "python",
    "scripts/fetch_benchmark_data.py",
    "--benchmark",
    "superni",
    "--destination",
    PROJECT / "data" / "SAPT",
)
training_arguments = [
    "python",
    "-m",
    "slao_repro.train",
    "--config",
    "configs/paper/qwen25_3b_superni_o1.yaml",
    "--method",
    METHOD,
    "--seed",
    SEED,
    "--run-label",
    RUN_ID,
    "--output-dir",
    ARTIFACT_ROOT / "training",
    "--resume-checkpoint",
    checkpoint_path,
    "--resume-metrics",
    metrics_path,
]
if predictions_path.is_file():
    training_arguments.extend(["--resume-predictions", predictions_path])
if STOP_AFTER_TASK_INDEX:
    training_arguments.extend(["--stop-after-task-index", STOP_AFTER_TASK_INDEX])
run_uv(*training_arguments)
