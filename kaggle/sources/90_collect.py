# ruff: noqa: F821
"""Fail-fast result verification and compact downloadable index."""

summary_path = ARTIFACT_ROOT / "training" / "summary.json"
manifest_path = ARTIFACT_ROOT / "training" / "manifest.json"
metrics_path = ARTIFACT_ROOT / "training" / "metrics.jsonl"
for required_path in (summary_path, manifest_path, metrics_path):
    if not required_path.is_file() or required_path.stat().st_size == 0:
        raise RuntimeError(f"required result artifact missing or empty: {required_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if summary.get("status") != "completed":
    raise RuntimeError(f"training summary is not completed: {summary.get('status')}")
if manifest.get("git_revision") != EXPECTED_SHA:
    raise RuntimeError("training manifest commit differs from staged commit")
index = {
    "status": "verified",
    "run_id": RUN_ID,
    "git_revision": EXPECTED_SHA,
    "method": summary.get("method"),
    "seed": summary.get("seed"),
    "tasks_completed": summary.get("tasks_completed"),
    "aa_percent": summary.get("aa_percent"),
    "bwt_percentage_points": summary.get("bwt_percentage_points"),
    "paper_comparison": summary.get("paper_comparison"),
    "runtime_s": summary.get("runtime_s"),
    "hardware": summary.get("hardware"),
    "paths": {
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "metrics": str(metrics_path),
    },
}
(ARTIFACT_ROOT / "kaggle_result_index.json").write_text(
    json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("SLAO_KAGGLE_RESULT " + json.dumps(index, sort_keys=True))
