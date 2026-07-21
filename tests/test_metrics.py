import pytest

from slao_repro.metrics import aa, aopd, bwt, exact_match, mopd, task_metric


def test_aa_and_bwt_match_paper_definitions() -> None:
    matrix = [
        [80.0, None, None],
        [75.0, 70.0, None],
        [72.0, 68.0, 90.0],
    ]
    assert aa([72.0, 68.0, 90.0]) == pytest.approx(76.6666667)
    assert bwt(matrix) == pytest.approx(-5.0)


def test_mopd_and_aopd_are_taskwise_order_ranges() -> None:
    orders = [[80.0, 60.0, 50.0], [78.0, 67.0, 49.0], [81.0, 63.0, 55.0]]
    assert mopd(orders) == pytest.approx(7.0)
    assert aopd(orders) == pytest.approx((3.0 + 7.0 + 6.0) / 3.0)


def test_classification_normalization_and_task_metric() -> None:
    assert exact_match(" Positive! ", ["positive"]) == 1.0
    score = task_metric(["yes", "NO."], [("Yes",), ("no",)], classification=True)
    assert score == 100.0
