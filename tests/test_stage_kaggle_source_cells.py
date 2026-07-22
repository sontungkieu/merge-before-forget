from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stage_kaggle_source_cells", Path("scripts/stage_kaggle_source_cells.py")
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
render_sources = MODULE.render_sources


def test_render_paper_sources_replaces_all_placeholders(tmp_path: Path) -> None:
    rendered = render_sources(
        source_dir=Path("kaggle/sources"),
        output_dir=tmp_path / "sources",
        mode="paper",
        expected_sha="a" * 40,
        run_id="paper-retry",
        method="slao",
        seed=42,
    )

    assert [path.name for path in rendered] == ["00_setup.py", "10_paper.py", "90_collect.py"]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in rendered)
    assert "__EXPECTED_SHA__" not in combined
    assert "__RUN_ID__" not in combined
    assert 'METHOD = \'slao\'' in combined
    assert "SEED = '42'" in combined
    assert "a" * 40 in combined
    assert "paper-retry" in combined


def test_render_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "sources"
    output.mkdir()
    (output / "owned.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        render_sources(
            source_dir=Path("kaggle/sources"),
            output_dir=output,
            mode="paper",
            expected_sha="b" * 40,
            run_id="paper-retry",
            method="slao",
            seed=42,
        )
