"""Fetch a pinned sparse copy of the official SAPT SuperNI benchmark data."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SAPT_REPOSITORY = "https://github.com/circle-hit/SAPT.git"
SAPT_COMMIT = "52a52b920324c656bdb6dac08e43dc600ba22f21"


def _run(*command: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({error.returncode}): {rendered}\n{detail}"
        ) from error
    return result.stdout.strip()


def _task_files(destination: Path) -> list[Path]:
    return sorted((destination / "CL_Benchmark/SuperNI").glob("*/*.json"))


def _checkout_is_ready(destination: Path) -> bool:
    if not (destination / ".git").is_dir():
        return False
    actual = _run("git", "rev-parse", "HEAD", cwd=destination)
    return actual == SAPT_COMMIT and len(_task_files(destination)) == 45


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["superni"], required=True)
    parser.add_argument("--destination", default="data/SAPT")
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    if destination.exists() and not (destination / ".git").is_dir():
        raise FileExistsError(f"destination exists but is not a git checkout: {destination}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            SAPT_REPOSITORY,
            str(destination),
        )
        _run("git", "sparse-checkout", "init", "--cone", cwd=destination)
        _run(
            "git",
            "sparse-checkout",
            "set",
            "CL_Benchmark/SuperNI",
            "configs/SuperNI",
            cwd=destination,
        )
    fetch_action = "reused"
    if not _checkout_is_ready(destination):
        _run("git", "fetch", "--depth", "1", "origin", SAPT_COMMIT, cwd=destination)
        _run("git", "checkout", "--detach", SAPT_COMMIT, cwd=destination)
        fetch_action = "fetched"
    actual = _run("git", "rev-parse", "HEAD", cwd=destination)
    if actual != SAPT_COMMIT:
        raise RuntimeError(f"SAPT checkout drift: {actual} != {SAPT_COMMIT}")
    task_files = _task_files(destination)
    if len(task_files) != 45:
        raise RuntimeError(f"expected 45 SuperNI split files, found {len(task_files)}")
    split_counts = {
        str(path.relative_to(destination / "CL_Benchmark/SuperNI")): len(
            json.loads(path.read_text(encoding="utf-8")).get("Instances", [])
        )
        for path in task_files
    }
    manifest = {
        "benchmark": args.benchmark,
        "repository": SAPT_REPOSITORY,
        "commit": actual,
        "fetch_action": fetch_action,
        "split_file_count": len(task_files),
        "split_counts": split_counts,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = destination / "slao_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}))


if __name__ == "__main__":
    main()
