# ruff: noqa: F821
"""Single-seed, full SuperNI O1 paper-cell or matched baseline run."""

METHOD = "__METHOD__"
SEED = "__SEED__"

if METHOD not in {"slao", "seqlora", "ftba_mb"}:
    raise ValueError(f"unsupported staged method: {METHOD}")
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
run_uv(
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
)
