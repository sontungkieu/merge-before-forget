import ast
import importlib.util
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).parents[1] / "kaggle" / "tpu" / "sources"
PYTHON_SOURCE = SOURCE_ROOT / "30_transformers_peft_xla_compat.py"
CELL_SOURCE = SOURCE_ROOT / "30_transformers_peft_xla_compat.cell"


def load_compat_module():
    spec = importlib.util.spec_from_file_location("tpu_transformers_peft_compat", PYTHON_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compatibility_source_is_valid_python_and_imports_without_ml_runtime():
    source = PYTHON_SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    module = load_compat_module()

    assert module.EXPECTED_TORCH_RELEASE == "2.8.0"
    assert module.EXPECTED_TORCH_XLA_RELEASE == "2.8.0"
    assert module.EXPECTED_TRANSFORMERS_RELEASE == "4.51.3"
    assert module.EXPECTED_PEFT_RELEASE == "0.15.2"
    assert module.EXPECTED_TPU_DEVICE_COUNT == 8


def test_release_version_normalizes_local_suffix_and_rejects_empty_input():
    module = load_compat_module()

    assert module.release_version("2.8.0+cpu") == "2.8.0"
    assert module.release_version("4.51.3") == "4.51.3"
    with pytest.raises(ValueError, match="must not be empty"):
        module.release_version("  ")


def test_gate_rejects_untracked_invocations_before_importing_ml_runtime(monkeypatch):
    module = load_compat_module()
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    monkeypatch.delenv("SLAO_TPU_SOURCE_COMMIT", raising=False)

    with pytest.raises(RuntimeError, match="through uv run"):
        module.run_compatibility_gate()

    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    with pytest.raises(RuntimeError, match="exact 40-hex Git commit"):
        module.run_compatibility_gate()


def test_compatibility_cell_requires_an_exact_commit_and_preserves_xla_runtime():
    source = CELL_SOURCE.read_text(encoding="utf-8")

    assert "SOURCE_COMMIT_TO_PIN_BEFORE_SUBMIT" in source
    assert "^[0-9a-f]{40}$" in source
    assert "SLAO_TPU_SOURCE_COMMIT" in source
    assert "uv run --no-project" in source
    assert "torch==" not in source
    assert "torch_xla==" not in source
    assert "kaggle kernels push" not in source
