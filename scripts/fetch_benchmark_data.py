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
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


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
    _run("git", "fetch", "--depth", "1", "origin", SAPT_COMMIT, cwd=destination)
    _run("git", "checkout", "--detach", SAPT_COMMIT, cwd=destination)
    actual = _run("git", "rev-parse", "HEAD", cwd=destination)
    if actual != SAPT_COMMIT:
        raise RuntimeError(f"SAPT checkout drift: {actual} != {SAPT_COMMIT}")
    task_files = sorted((destination / "CL_Benchmark/SuperNI").glob("*/*.json"))
    if len(task_files) != 45:
        raise RuntimeError(f"expected 45 SuperNI split files, found {len(task_files)}")
    manifest = {
        "benchmark": args.benchmark,
        "repository": SAPT_REPOSITORY,
        "commit": actual,
        "split_file_count": len(task_files),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = destination / "slao_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}))


if __name__ == "__main__":
    main()

