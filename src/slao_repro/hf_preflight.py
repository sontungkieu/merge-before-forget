"""Fail-closed Hugging Face identity and gated-checkpoint access preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download, whoami


def _model_provenance(config_path: Path) -> tuple[str, str]:
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.model must be a mapping")
    model_id = model.get("id")
    revision = model.get("revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("config.model.id must be a non-empty string")
    if not isinstance(revision, str) or not revision:
        raise ValueError("config.model.revision must be a non-empty string")
    return model_id, revision


def verify_hugging_face_access(
    *,
    config_path: Path,
    expected_account: str,
    token: str,
) -> dict[str, str]:
    identity = whoami(token=token)
    actual_account = identity.get("name")
    if actual_account != expected_account:
        raise RuntimeError(
            f"Hugging Face identity mismatch: expected={expected_account!r}, "
            f"actual={actual_account!r}"
        )

    model_id, revision = _model_provenance(config_path)
    hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        revision=revision,
        token=token,
    )
    return {
        "status": "ok",
        "account": actual_account,
        "model_id": model_id,
        "revision": revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    expected_account = os.environ.get("HF_EXPECTED_ACCOUNT")
    if not token:
        raise RuntimeError("HF_TOKEN is not loaded from the selected Talapas profile")
    if not expected_account:
        raise RuntimeError("HF_EXPECTED_ACCOUNT is not loaded from the selected profile")

    result = verify_hugging_face_access(
        config_path=args.config,
        expected_account=expected_account,
        token=token,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
