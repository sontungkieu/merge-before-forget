"""PEFT adapter state extraction and restoration."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from slao_repro.slao import AdapterState


def iter_lora_layers(
    model: nn.Module, adapter_name: str = "default"
) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        try:
            a_layer = lora_a[adapter_name]
            b_layer = lora_b[adapter_name]
        except (KeyError, TypeError):
            continue
        if hasattr(a_layer, "weight") and hasattr(b_layer, "weight"):
            yield name, module


def capture_adapter_state(model: nn.Module, adapter_name: str = "default") -> AdapterState:
    state: AdapterState = {}
    for name, module in iter_lora_layers(model, adapter_name):
        state[name] = {
            "A": module.lora_A[adapter_name].weight.detach().cpu().float().clone(),
            "B": module.lora_B[adapter_name].weight.detach().cpu().float().clone(),
        }
    if not state:
        raise ValueError("model exposes no PEFT LoRA A/B layers")
    return state


def set_adapter_state(model: nn.Module, state: AdapterState, adapter_name: str = "default") -> None:
    layers = dict(iter_lora_layers(model, adapter_name))
    if layers.keys() != state.keys():
        missing = sorted(set(layers) - set(state))
        extra = sorted(set(state) - set(layers))
        raise ValueError(f"adapter layer mismatch; missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, module in layers.items():
            for part, container_name in (("A", "lora_A"), ("B", "lora_B")):
                parameter = getattr(module, container_name)[adapter_name].weight
                source = state[name][part]
                if parameter.shape != source.shape:
                    raise ValueError(
                        f"shape mismatch for {name}.{part}: {parameter.shape} != {source.shape}"
                    )
                parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))


def adapter_parameters(model: nn.Module, adapter_name: str = "default") -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for _, module in iter_lora_layers(model, adapter_name):
        parameters.extend(
            [module.lora_A[adapter_name].weight, module.lora_B[adapter_name].weight]
        )
    if not parameters:
        raise ValueError("model exposes no trainable LoRA parameters")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in parameters:
        parameter.requires_grad = True
    return parameters
