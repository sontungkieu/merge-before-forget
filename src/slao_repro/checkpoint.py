"""Small adapter-only checkpoint contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from slao_repro.slao import SLAOMerger


def save_checkpoint(path: str | Path, merger: SLAOMerger, metadata: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "slao-repro-v1",
        "merger": merger.state_dict(),
        "metadata": metadata,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(path: str | Path) -> tuple[SLAOMerger, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "slao-repro-v1":
        raise ValueError("unsupported SLAO checkpoint format")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a dictionary")
    return SLAOMerger.from_state_dict(payload["merger"]), metadata

