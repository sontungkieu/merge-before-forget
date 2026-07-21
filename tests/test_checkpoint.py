import torch

from slao_repro.checkpoint import load_checkpoint, save_checkpoint
from slao_repro.slao import SLAOMerger


def test_checkpoint_roundtrip(tmp_path) -> None:
    merger = SLAOMerger()
    original = {
        "layer": {
            "A": torch.randn(2, 5, generator=torch.Generator().manual_seed(1)),
            "B": torch.randn(4, 2, generator=torch.Generator().manual_seed(2)),
        }
    }
    merger.add_first_task(original)
    path = tmp_path / "adapter.pt"
    save_checkpoint(path, merger, {"seed": 42, "task": "toy"})
    restored, metadata = load_checkpoint(path)
    assert restored.task_index == 1
    assert metadata == {"seed": 42, "task": "toy"}
    torch.testing.assert_close(restored.merged["layer"]["A"], original["layer"]["A"])
    torch.testing.assert_close(restored.merged["layer"]["B"], original["layer"]["B"])

