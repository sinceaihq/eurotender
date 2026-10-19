from __future__ import annotations

import numpy as np
import pytest

from eurotender.metrics import (
    classification_report,
    expected_calibration_error,
    per_class_f1,
    top_k_accuracy,
)


def test_top_k_accuracy() -> None:
    y = np.array(["a", "b", "c"])
    ranked = np.array([["a", "b"], ["a", "b"], ["a", "b"]])
    assert top_k_accuracy(y, ranked, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(y, ranked, 2) == pytest.approx(2 / 3)
    assert np.isnan(top_k_accuracy(np.array([]), np.empty((0, 2)), 1))


def test_report_perfect() -> None:
    y = ["a", "b", "b"]
    ranked = np.array([["a", "b"], ["b", "a"], ["b", "a"]])
    r = classification_report(y, ranked, ks=(1, 2, 5))
    assert r["top1_accuracy"] == 1.0
    assert r["macro_f1"] == 1.0
    assert r["weighted_f1"] == 1.0
    assert "top5_accuracy" not in r  # only 2 columns ranked


def test_report_macro_vs_weighted() -> None:
    # Majority predictor: right on 9 of 10, never predicts the minority class.
    y = ["a"] * 9 + ["b"]
    ranked = np.array([["a", "b"]] * 10)
    r = classification_report(y, ranked)
    assert r["top1_accuracy"] == pytest.approx(0.9)
    assert r["macro_f1"] < r["weighted_f1"]
    assert r["macro_recall"] == pytest.approx(0.5)


def test_per_class_f1() -> None:
    out = per_class_f1(["a", "a", "b"], ["a", "b", "b"])
    assert out["a"]["recall"] == 0.5
    assert out["b"]["precision"] == 0.5
    assert out["a"]["support"] == 2


def test_ece() -> None:
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0
    assert expected_calibration_error(np.array([0.9, 0.9]), np.array([0, 0])) == pytest.approx(0.9)
    assert np.isnan(expected_calibration_error(np.array([]), np.array([])))
