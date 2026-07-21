import json
from pathlib import Path

import pytest

from slao_repro.prepare_model import _weight_files


def test_weight_files_verifies_every_index_shard(tmp_path: Path) -> None:
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    assert [path.name for path in _weight_files(tmp_path)] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_weight_files_rejects_missing_referenced_shard(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "missing.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing referenced shards"):
        _weight_files(tmp_path)
