"""Continual-learning and task metrics used by the SLAO paper."""

from __future__ import annotations

import string
from collections.abc import Sequence

from rouge_score import rouge_scorer


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    return " ".join(without_punctuation.split())


def exact_match(prediction: str, references: Sequence[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(reference) for reference in references))


def rouge_l(prediction: str, references: Sequence[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return max(
        scorer.score(target=reference, prediction=prediction)["rougeL"].fmeasure
        for reference in references
    )


def task_metric(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    classification: bool,
) -> float:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and non-empty references must have equal length")
    metric = exact_match if classification else rouge_l
    return 100.0 * sum(
        metric(pred, refs) for pred, refs in zip(predictions, references, strict=True)
    ) / len(
        predictions
    )


def aa(final_scores: Sequence[float]) -> float:
    if not final_scores:
        raise ValueError("AA requires at least one task score")
    return sum(float(value) for value in final_scores) / len(final_scores)


def bwt(score_matrix: Sequence[Sequence[float | None]]) -> float:
    """Compute paper BWT from a lower-triangular task-by-time matrix."""

    task_count = len(score_matrix)
    if task_count < 2:
        raise ValueError("BWT requires at least two tasks")
    final = score_matrix[-1]
    if len(final) < task_count:
        raise ValueError("final score row is incomplete")
    deltas = []
    for task_index in range(task_count - 1):
        diagonal = score_matrix[task_index][task_index]
        final_value = final[task_index]
        if diagonal is None or final_value is None:
            raise ValueError("BWT requires diagonal and final scores")
        deltas.append(float(final_value) - float(diagonal))
    return sum(deltas) / (task_count - 1)


def _opd_by_task(order_scores: Sequence[Sequence[float]]) -> list[float]:
    if len(order_scores) < 2:
        raise ValueError("order disparity requires at least two task orders")
    task_count = len(order_scores[0])
    if task_count == 0 or any(len(row) != task_count for row in order_scores):
        raise ValueError("all order-score rows must have the same non-zero length")
    return [
        max(float(row[task]) for row in order_scores)
        - min(float(row[task]) for row in order_scores)
        for task in range(task_count)
    ]


def mopd(order_scores: Sequence[Sequence[float]]) -> float:
    return max(_opd_by_task(order_scores))


def aopd(order_scores: Sequence[Sequence[float]]) -> float:
    values = _opd_by_task(order_scores)
    return sum(values) / len(values)


def symmetric_relative_delta(observed: float, target: float, epsilon: float = 1e-12) -> float:
    return abs(observed - target) / max(abs(observed), abs(target), epsilon)
