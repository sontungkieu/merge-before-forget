import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/fetch_benchmark_data.py"
SPEC = importlib.util.spec_from_file_location("fetch_benchmark_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
fetch_benchmark_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_benchmark_data)


def _make_checkout(root: Path, split_file_count: int) -> Path:
    (root / ".git").mkdir(parents=True)
    split_root = root / "CL_Benchmark/SuperNI"
    for index in range(split_file_count):
        task_dir = split_root / f"task_{index:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "train.json").write_text('{"Instances": []}\n')
    return root


@pytest.mark.parametrize(
    ("revision", "split_file_count", "expected"),
    [
        (fetch_benchmark_data.SAPT_COMMIT, 45, True),
        (fetch_benchmark_data.SAPT_COMMIT, 44, False),
        ("0" * 40, 45, False),
    ],
)
def test_checkout_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
    split_file_count: int,
    expected: bool,
) -> None:
    checkout = _make_checkout(tmp_path / "SAPT", split_file_count)
    monkeypatch.setattr(fetch_benchmark_data, "_run", lambda *args, **kwargs: revision)

    assert fetch_benchmark_data._checkout_is_ready(checkout) is expected
