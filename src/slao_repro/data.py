"""Pinned SAPT/SuperNI data reader and prompt formatter."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuperNIExample:
    task: str
    prompt: str
    references: tuple[str, ...]
    target: str
    example_id: str


def _prompt(definition: str, input_text: str) -> str:
    return (
        f"Definition: {definition.strip()}\n\n"
        "Now complete the following example -\n"
        f"Input: {input_text.strip()}\n"
        "Output: "
    )


def load_superni_task(
    root: str | Path,
    task: str,
    split: str,
    *,
    max_samples: int | None,
    seed: int,
) -> list[SuperNIExample]:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"unsupported SuperNI split: {split}")
    path = Path(root) / task / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing SuperNI split: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    definitions = raw.get("Definition") or []
    definition = definitions[0] if isinstance(definitions, list) else definitions
    instances = list(raw.get("Instances") or [])
    if max_samples is not None:
        instances = instances[:max_samples]
    rng = random.Random(seed)
    examples: list[SuperNIExample] = []
    for index, instance in enumerate(instances):
        outputs = instance.get("output")
        if isinstance(outputs, str):
            references = (outputs.strip(),)
        else:
            references = tuple(str(output).strip() for output in outputs or [])
        if not references:
            raise ValueError(f"{path} instance {index} has no outputs")
        examples.append(
            SuperNIExample(
                task=task,
                prompt=_prompt(str(definition), str(instance["input"])),
                references=references,
                target=references[rng.randrange(len(references))],
                example_id=str(instance.get("id", index)),
            )
        )
    if not examples:
        raise ValueError(f"{path} yielded no examples")
    return examples

