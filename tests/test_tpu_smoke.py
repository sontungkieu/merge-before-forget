import importlib.util
from pathlib import Path

import pytest
import torch


def load_smoke_module():
    source = (
        Path(__file__).parents[1]
        / "kaggle"
        / "tpu"
        / "sources"
        / "20_torch_xla_lora_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("tpu_lora_smoke", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lora_smoke_layer_freezes_base_and_initializes_b_to_zero():
    module = load_smoke_module()
    layer = module.LoRALinear(8, 4, 2)

    assert layer.weight.requires_grad is False
    assert layer.lora_A.requires_grad is True
    assert layer.lora_B.requires_grad is True
    assert torch.count_nonzero(layer.lora_A) > 0
    assert torch.count_nonzero(layer.lora_B) == 0
    assert [name for name, value in layer.named_parameters() if value.requires_grad] == [
        "lora_A",
        "lora_B",
    ]


def test_lora_smoke_rejects_invalid_rank():
    module = load_smoke_module()
    with pytest.raises(ValueError, match="rank"):
        module.LoRALinear(8, 4, 5)
