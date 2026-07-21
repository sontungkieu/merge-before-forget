"""Kaggle setup cell; placeholders are filled only in the staged copy."""

import json
import os
import subprocess
from pathlib import Path

EXPECTED_SHA = "__EXPECTED_SHA__"
RUN_ID = "__RUN_ID__"
REPOSITORY = "https://github.com/sontungkieu/merge-before-forget.git"
PROJECT = Path("/kaggle/working/merge-before-forget")
WORK_ROOT = Path("/kaggle/working/slao-runtime")
UV_VERSION = "0.10.2"
UV_BIN = WORK_ROOT / "tools" / "uv"
ENVIRONMENT = WORK_ROOT / "venv"
ARTIFACT_ROOT = Path("/kaggle/working/slao-artifacts") / RUN_ID


def run(command, *, cwd=None):
    print("SLAO_COMMAND " + json.dumps([str(item) for item in command]))
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


if PROJECT.exists() or ARTIFACT_ROOT.exists():
    raise RuntimeError("fresh Kaggle paths are unexpectedly non-empty")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
(WORK_ROOT / "tools").mkdir(parents=True, exist_ok=True)
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)

run(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, PROJECT])
run(["git", "checkout", "--detach", EXPECTED_SHA], cwd=PROJECT)
actual_sha = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, check=True, capture_output=True, text=True
).stdout.strip()
if actual_sha != EXPECTED_SHA:
    raise RuntimeError(f"checkout mismatch: {actual_sha} != {EXPECTED_SHA}")

archive = WORK_ROOT / f"uv-{UV_VERSION}.tar.gz"
run(
    [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "3",
        f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz",
        "--output",
        archive,
    ]
)
run(["tar", "-xzf", archive, "-C", WORK_ROOT / "tools"])
run(["install", "-m", "0755", WORK_ROOT / "tools" / "uv-x86_64-unknown-linux-gnu" / "uv", UV_BIN])

os.environ["UV_PROJECT_ENVIRONMENT"] = str(ENVIRONMENT)
os.environ["UV_CACHE_DIR"] = str(WORK_ROOT / "uv-cache")
os.environ["HF_HOME"] = str(WORK_ROOT / "hf")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
run([UV_BIN, "sync", "--project", PROJECT, "--frozen", "--extra", "test"])


def run_uv(*arguments):
    run([UV_BIN, "run", "--project", PROJECT, "--no-sync", *arguments], cwd=PROJECT)


setup_summary = {
    "status": "completed",
    "run_id": RUN_ID,
    "repository": REPOSITORY,
    "git_revision": actual_sha,
    "uv_version": subprocess.run(
        [UV_BIN, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip(),
    "project": str(PROJECT),
    "environment": str(ENVIRONMENT),
    "artifact_root": str(ARTIFACT_ROOT),
    "secret_mode": "none",
}
(ARTIFACT_ROOT / "setup_summary.json").write_text(
    json.dumps(setup_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("SLAO_SETUP_SUMMARY " + json.dumps(setup_summary, sort_keys=True))
