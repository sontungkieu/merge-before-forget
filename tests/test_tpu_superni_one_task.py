from __future__ import annotations

import ast
import re
from pathlib import Path

CELL_SOURCE = Path(__file__).parents[1] / "kaggle" / "tpu" / "sources" / "40_superni_one_task.cell"


def test_one_task_cell_preserves_scientific_hyperparameters() -> None:
    source = CELL_SOURCE.read_text(encoding="utf-8")

    assert "configs/paper/qwen25_3b_superni_o1.yaml" in source
    assert "--method slao" in source
    assert "--seed 42" in source
    assert "--runtime xla" in source
    assert "--max-tasks 1" in source
    assert "--max-train-samples" not in source
    assert "--max-eval-samples" not in source
    assert "--max-optimizer-steps-per-task" not in source
    assert re.search(r"(?m)^\\s*--epochs(?:\\s|$)", source) is None


def test_one_task_cell_provisions_without_shadowing_torch_xla() -> None:
    source = CELL_SOURCE.read_text(encoding="utf-8")

    assert "--no-deps" in source
    assert '--target "${dependency_root}"' in source
    for requirement in (
        "transformers==4.51.3",
        "peft==0.15.2",
        "accelerate==1.6.0",
        "safetensors==0.5.3",
        "tokenizers==0.21.4",
        "huggingface-hub==0.36.2",
        "pyyaml==6.0.2",
        "rouge-score==0.1.2",
    ):
        assert f"'{requirement}'" in source
    assert "'torch==" not in source
    assert "'torch_xla==" not in source
    assert "protected package {name} is shadowed" in source
    assert "verify_runtime before" in source
    assert "verify_runtime after" in source
    assert "uv run --no-project" in source


def test_one_task_cell_requires_staged_commit_and_run_id() -> None:
    source = CELL_SOURCE.read_text(encoding="utf-8")

    assert "source_commit='__SOURCE_COMMIT__'" in source
    assert "run_id='__RUN_ID__'" in source
    assert "^[0-9a-f]{40}$" in source
    assert 'rev-parse HEAD)" = "${source_commit}"' in source
    assert "SLAO_TPU_SOURCE_COMMIT" in source


def test_one_task_cell_requires_runtime_checkpoint_and_audit_evidence() -> None:
    source = CELL_SOURCE.read_text(encoding="utf-8")

    assert "approximate_portability_one_task_superni" in source
    assert '"scientific_result": False' in source
    assert '"paper_comparable": False' in source
    assert 'hardware.get("xla_visible_device_count") != 8' in source
    assert 'hardware.get("uses_all_visible_devices")' in source
    assert "first_optimizer_step_s" in source
    assert "runtime_memory_before" in source
    assert "runtime_memory_after" in source
    assert "cpu_portable" in source
    assert "SLAO_TPU_SUPERNI_ONE_TASK" in source


def test_one_task_cell_contains_no_submit_or_secret_logic() -> None:
    source = CELL_SOURCE.read_text(encoding="utf-8")

    assert "kaggle kernels push" not in source
    assert "KAGGLE_KEY" not in source
    assert "KAGGLE_USERNAME" not in source
    assert "HF_TOKEN" not in source


def test_one_task_cell_embedded_python_is_valid() -> None:
    source = CELL_SOURCE.read_text(encoding="utf-8")
    blocks = source.split("<<'PY'\n")[1:]

    assert len(blocks) == 3
    for block in blocks:
        python_source, separator, _rest = block.partition("\nPY\n")
        assert separator
        ast.parse(python_source)
