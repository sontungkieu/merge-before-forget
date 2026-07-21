# ruff: noqa: F821
"""Static, synthetic, overfit, sequential, and tiny Qwen integration gates."""

run_uv("ruff", "check", ".")
run_uv("pytest", "-q", "-s")
run_uv(
    "python",
    "scripts/dev_gates.py",
    "--output-dir",
    ARTIFACT_ROOT / "development",
)
run_uv(
    "python",
    "scripts/fetch_benchmark_data.py",
    "--benchmark",
    "superni",
    "--destination",
    PROJECT / "data" / "SAPT",
)
run_uv(
    "python",
    "-m",
    "slao_repro.train",
    "--config",
    "configs/paper/qwen25_3b_superni_o1.yaml",
    "--method",
    "slao",
    "--seed",
    "42",
    "--run-label",
    RUN_ID,
    "--output-dir",
    ARTIFACT_ROOT / "training",
    "--max-tasks",
    "1",
    "--max-train-samples",
    "8",
    "--max-eval-samples",
    "4",
    "--epochs",
    "1",
    "--max-optimizer-steps-per-task",
    "2",
)
