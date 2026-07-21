"""Download and verify every referenced model weight shard before training."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _weight_files(snapshot: Path) -> list[Path]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid or empty weight map: {index_path}")
        names = sorted(set(weight_map.values()))
        paths = [snapshot / str(name) for name in names]
    elif (snapshot / "model.safetensors").is_file():
        paths = [snapshot / "model.safetensors"]
    else:
        raise FileNotFoundError("model snapshot has neither safetensors index nor single weights")
    missing = [str(path.relative_to(snapshot)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model snapshot is missing referenced shards: {missing}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be positive")
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_id = str(config["model"]["id"])
    revision = str(config["model"]["revision"])
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()
    for attempt in range(1, args.attempts + 1):
        attempt_started = time.monotonic()
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    max_workers=args.max_workers,
                )
            )
            required = [snapshot / name for name in ("config.json", "tokenizer_config.json")]
            missing_metadata = [path.name for path in required if not path.is_file()]
            if missing_metadata:
                raise FileNotFoundError(f"model snapshot is missing metadata: {missing_metadata}")
            weights = _weight_files(snapshot)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "completed",
                    "elapsed_s": time.monotonic() - attempt_started,
                }
            )
            result = {
                "status": "completed",
                "at_utc": _utc_now(),
                "model_id": model_id,
                "revision": revision,
                "snapshot_path": str(snapshot),
                "weight_files": [path.name for path in weights],
                "weight_bytes": sum(path.stat().st_size for path in weights),
                "attempts": attempts,
                "elapsed_s": time.monotonic() - started,
                "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET"),
            }
            _write_json(output_path, result)
            print("SLAO_MODEL_PREPARE " + json.dumps(result, sort_keys=True), flush=True)
            return
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "elapsed_s": time.monotonic() - attempt_started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print("SLAO_MODEL_PREPARE_ATTEMPT " + json.dumps(attempts[-1]), flush=True)
    failure = {
        "status": "failed",
        "at_utc": _utc_now(),
        "model_id": model_id,
        "revision": revision,
        "attempts": attempts,
        "elapsed_s": time.monotonic() - started,
        "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET"),
    }
    _write_json(output_path, failure)
    raise RuntimeError(f"model preparation failed after {args.attempts} attempts")


if __name__ == "__main__":
    main()
