"""Evaluation metrics for ranked CPV predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def top_k_accuracy(y_true: np.ndarray, ranked: np.ndarray, k: int) -> float:
    """Share of rows whose true label is among the first ``k`` ranked labels."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.any(ranked[:, :k] == y_true[:, None], axis=1)))


def classification_report(
    y_true: Sequence[str] | np.ndarray, ranked: np.ndarray, ks: Sequence[int] = (1, 3, 5)
) -> dict[str, Any]:
    """Accuracy, top-k, and macro / weighted precision, recall and F1.

    ``ranked`` is an ``(n, K)`` array of label strings, best first. Macro
    averages are taken over the labels present in ``y_true`` so that classes
    the model never sees in evaluation do not dilute the score.
    """
    y = np.asarray(y_true, dtype=object)
    ranked = np.asarray(ranked, dtype=object)
    pred = ranked[:, 0]
    labels = sorted(set(y))
    out: dict[str, Any] = {"n": len(y)}
    for k in ks:
        if k <= ranked.shape[1]:
            out[f"top{k}_accuracy"] = top_k_accuracy(y, ranked, k)
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y, pred, labels=labels, average=avg, zero_division=0
        )
        out[f"{avg}_precision"] = float(p)
        out[f"{avg}_recall"] = float(r)
        out[f"{avg}_f1"] = float(f)
    return out


def per_class_f1(y_true: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, float]]:
    labels = sorted(set(y_true))
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred, labels=labels, average=None, zero_division=0
    )
    return {
        lab: {"precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
        for lab, pi, ri, fi, si in zip(labels, p, r, f, s, strict=True)
    }


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """ECE of the top-1 confidence with equal-width bins."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if len(confidence) == 0:
        return float("nan")
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)
