from math import sqrt

import pytest
import torch

from slao_repro.slao import (
    MergedBInitSLAOMerger,
    SLAOMerger,
    canonical_qr_rows,
    initialize_from_last_finetuned,
    merge_finetuned_state,
    time_aware_coefficient,
)


def state(a: torch.Tensor, b: torch.Tensor):
    return {"layer": {"A": a.clone(), "B": b.clone()}}


def test_qr_rows_are_orthonormal_and_sign_canonical() -> None:
    previous = torch.tensor([[1.0, -2.0, 0.5, 1.0], [-3.0, 0.5, 2.0, -1.0]])
    initialized = canonical_qr_rows(previous)
    torch.testing.assert_close(initialized @ initialized.T, torch.eye(2), atol=1e-6, rtol=1e-6)
    q, r = torch.linalg.qr(previous.T, mode="reduced")
    signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    torch.testing.assert_close(initialized, (q * signs.unsqueeze(0)).T)
    assert torch.all(signs * torch.diagonal(r) >= 0)


def test_qr_zero_diagonal_does_not_zero_basis() -> None:
    previous = torch.zeros(2, 4)
    initialized = canonical_qr_rows(previous)
    torch.testing.assert_close(initialized @ initialized.T, torch.eye(2))


def test_ab_initialization_uses_qr_a_and_previous_finetuned_b() -> None:
    previous_a = torch.randn(3, 7, generator=torch.Generator().manual_seed(4))
    previous_b = torch.randn(5, 3, generator=torch.Generator().manual_seed(5))
    initialized = initialize_from_last_finetuned(state(previous_a, previous_b))
    torch.testing.assert_close(initialized["layer"]["B"], previous_b)
    torch.testing.assert_close(
        initialized["layer"]["A"] @ initialized["layer"]["A"].T,
        torch.eye(3),
        atol=1e-6,
        rtol=1e-6,
    )
    assert initialized["layer"]["B"].data_ptr() != previous_b.data_ptr()


@pytest.mark.parametrize("index,expected", [(1, 1.0), (2, 1 / sqrt(2)), (4, 0.5)])
def test_time_aware_coefficient(index: int, expected: float) -> None:
    assert time_aware_coefficient(index) == pytest.approx(expected)


def test_sequential_merge_replaces_a_and_interpolates_b() -> None:
    a1, b1 = torch.eye(2, 4), torch.ones(3, 2)
    a2, b2 = -torch.eye(2, 4), torch.full((3, 2), 3.0)
    merged = merge_finetuned_state(state(a1, b1), state(a2, b2), 2)
    torch.testing.assert_close(merged["layer"]["A"], a2)
    torch.testing.assert_close(
        merged["layer"]["B"], b1 + (b2 - b1) / sqrt(2), atol=1e-6, rtol=1e-6
    )


def test_merger_initializes_from_last_finetuned_not_merged() -> None:
    merger = SLAOMerger()
    first = state(torch.randn(2, 4), torch.ones(3, 2))
    second = state(torch.randn(2, 4), torch.full((3, 2), 4.0))
    merger.add_first_task(first)
    merger.add_task(second)
    initialized = merger.next_initial_state()
    torch.testing.assert_close(initialized["layer"]["B"], second["layer"]["B"])
    assert not torch.allclose(initialized["layer"]["B"], merger.merged["layer"]["B"])


def test_third_party_variant_initializes_b_from_merged_state() -> None:
    merger = MergedBInitSLAOMerger()
    first = state(torch.eye(2, 4), torch.zeros(3, 2))
    second = state(-torch.eye(2, 4), torch.full((3, 2), 2.0))
    merger.add_first_task(first)
    merger.add_task(second)

    initialized = merger.next_initial_state()

    torch.testing.assert_close(initialized["layer"]["B"], merger.merged["layer"]["B"])
    assert not torch.allclose(initialized["layer"]["B"], second["layer"]["B"])
