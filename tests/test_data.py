import json

from slao_repro.data import load_superni_task


def test_superni_prompt_and_deterministic_target(tmp_path) -> None:
    task_dir = tmp_path / "task001"
    task_dir.mkdir()
    payload = {
        "Definition": ["Return a label."],
        "Instances": [
            {"id": "x", "input": "Example", "output": ["one", "two"]},
        ],
    }
    (task_dir / "train.json").write_text(json.dumps(payload), encoding="utf-8")
    first = load_superni_task(tmp_path, "task001", "train", max_samples=1, seed=5)
    second = load_superni_task(tmp_path, "task001", "train", max_samples=1, seed=5)
    assert first == second
    assert first[0].prompt == (
        "Definition: Return a label.\n\n"
        "Now complete the following example -\nInput: Example\nOutput: "
    )
    assert first[0].references == ("one", "two")

