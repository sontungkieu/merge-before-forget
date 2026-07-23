import json
from argparse import Namespace

import pytest
import torch

from slao_repro.train import _load_resume_inputs


def _write_resume_pair(tmp_path, *, method: str = "slao", seed: int = 42):
    task_order = ["task_a", "task_b"]
    checkpoint = tmp_path / "adapter_checkpoint.pt"
    metrics = tmp_path / "metrics.jsonl"
    torch.save(
        {
            "format": "slao-repro-adapter-v2",
            "method": method,
            "seed": seed,
            "task_index": 1,
            "task_order": task_order,
            "config_sha256": "config-sha",
            "fine_tuned": {"layer": {"A": torch.eye(2), "B": torch.eye(2)}},
            "score_matrix": [[10.0, None]],
            "merger": {
                "task_index": 1,
                "merged": {"layer": {"A": torch.eye(2), "B": torch.eye(2)}},
                "last_finetuned": {"layer": {"A": torch.eye(2), "B": torch.eye(2)}},
            },
        },
        checkpoint,
    )
    metrics.write_text(
        json.dumps(
            {
                "phase": "task_complete",
                "task_index": 1,
                "task": "task_a",
                "elapsed_s": 12.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return task_order, checkpoint, metrics


def test_resume_inputs_validate_and_preserve_elapsed_time(tmp_path) -> None:
    task_order, checkpoint, metrics = _write_resume_pair(tmp_path)
    args = Namespace(
        method="slao",
        seed=42,
        resume_checkpoint=str(checkpoint),
        resume_metrics=str(metrics),
        resume_predictions=None,
    )

    payload, records, elapsed_s, provenance = _load_resume_inputs(
        args, task_order, "config-sha"
    )

    assert payload["task_index"] == 1
    assert len(records) == 1
    assert elapsed_s == 12.5
    assert provenance["completed_tasks"] == 1


def test_resume_inputs_reject_method_mismatch(tmp_path) -> None:
    task_order, checkpoint, metrics = _write_resume_pair(tmp_path)
    args = Namespace(
        method="seqlora",
        seed=42,
        resume_checkpoint=str(checkpoint),
        resume_metrics=str(metrics),
        resume_predictions=None,
    )

    with pytest.raises(ValueError, match="method/seed"):
        _load_resume_inputs(args, task_order, "config-sha")


def test_resume_inputs_accept_legacy_v1_checkpoint(tmp_path) -> None:
    task_order, checkpoint, metrics = _write_resume_pair(
        tmp_path, method="seqlora"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["format"] = "slao-repro-adapter-v1"
    payload.pop("task_order")
    payload.pop("config_sha256")
    payload.pop("merger")
    torch.save(payload, checkpoint)
    args = Namespace(
        method="seqlora",
        seed=42,
        resume_checkpoint=str(checkpoint),
        resume_metrics=str(metrics),
        resume_predictions=None,
    )

    loaded, records, elapsed_s, _ = _load_resume_inputs(
        args, task_order, "config-sha"
    )

    assert loaded["format"] == "slao-repro-adapter-v1"
    assert len(records) == 1
    assert elapsed_s == 12.5
