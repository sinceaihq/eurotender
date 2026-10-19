"""Training utilities for the EuroTender CPV classifier.

Requires the ``train`` extra (pandas, pyarrow).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from .model import CpvModel, FeatureConfig, Featurizer, Head, softmax
from .text import compose

LEVEL_COLUMN = {
    "division": "cpv_division",
    "group": "cpv_group",
    "class": "cpv_class",
    "code": "cpv_main",
}


@dataclass
class TrainConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    levels: tuple[str, ...] = ("division", "group")
    classifier: str = "linear_svc"  # linear_svc | sgd_hinge | sgd_modified_huber | sgd_log | cnb
    c: float = 0.5  # LinearSVC regularisation
    alpha: float = 1e-6  # SGD regularisation
    nb_alpha: float = 0.1
    min_class_count: int = 10
    prune_threshold: float = 1e-3
    # Larger thresholds tried on the validation split; the largest one whose
    # top-1 accuracy stays within prune_tolerance of the unpruned head wins.
    prune_grid: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.08)
    prune_tolerance: float = 0.001
    max_train_rows: int | None = None
    seed: int = 20251019
    n_jobs: int = 4

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["levels"] = list(self.levels)
        d["prune_grid"] = list(self.prune_grid)
        return d


def load_split(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def texts_of(df: pd.DataFrame, max_description_chars: int) -> list[str]:
    return [
        compose(t, d, max_description_chars) for t, d in zip(df.title, df.description, strict=True)
    ]


def _make_estimator(cfg: TrainConfig) -> Any:
    if cfg.classifier == "linear_svc":
        svc = LinearSVC(
            C=cfg.c, loss="squared_hinge", dual=True, max_iter=200, tol=1e-3, random_state=cfg.seed
        )
        # liblinear is single-threaded; fit the one-vs-rest problems in parallel.
        return OneVsRestClassifier(svc, n_jobs=cfg.n_jobs)
    if cfg.classifier.startswith("sgd_"):
        loss = {
            "sgd_hinge": "hinge",
            "sgd_modified_huber": "modified_huber",
            "sgd_log": "log_loss",
        }[cfg.classifier]
        return SGDClassifier(
            loss=loss,
            alpha=cfg.alpha,
            max_iter=50,
            tol=1e-4,
            n_jobs=cfg.n_jobs,
            random_state=cfg.seed,
            average=False,
        )
    if cfg.classifier == "cnb":
        return ComplementNB(alpha=cfg.nb_alpha)
    raise ValueError(f"unknown classifier {cfg.classifier!r}")


def _to_head(level: str, estimator: Any, prune_threshold: float) -> Head:
    if isinstance(estimator, ComplementNB):
        # sklearn already stores the negated complement log-probabilities, so
        # the joint log-likelihood is X @ feature_log_prob_.T. Centring each
        # feature across classes only shifts every document's scores by a
        # constant, but makes the matrix far easier to prune.
        coef = estimator.feature_log_prob_
        coef = coef - coef.mean(axis=0, keepdims=True)
        intercept = np.zeros(coef.shape[0])
    elif isinstance(estimator, OneVsRestClassifier):
        coef = np.vstack([np.asarray(e.coef_).reshape(1, -1) for e in estimator.estimators_])
        intercept = np.concatenate([np.ravel(e.intercept_) for e in estimator.estimators_])
    else:
        coef = np.asarray(estimator.coef_)
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)
    classes = [str(c) for c in estimator.classes_]
    if coef.shape[0] == 1:  # binary problem: sklearn stores a single row
        coef = np.vstack([-coef[0], coef[0]])
        intercept = np.array([-intercept[0], intercept[0]])
    coef = coef.astype(np.float32)
    if prune_threshold > 0:
        coef[np.abs(coef) < prune_threshold] = 0.0
    weights = sp.csr_matrix(coef.T)
    return Head(level=level, classes=classes, weights=weights, intercept=intercept)


def fit_temperature(scores: np.ndarray, y_idx: np.ndarray) -> float:
    """Temperature that minimises validation NLL of softmax(scores / T)."""
    valid = y_idx >= 0
    if not valid.any():
        return 1.0
    s, y = scores[valid], y_idx[valid]
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.01), np.log(10.0), 121)):
        p = softmax(s / t)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def parallel_counts(
    featurizer: Featurizer, texts: Sequence[str], n_jobs: int, chunk: int = 20_000
) -> sp.csr_matrix:
    """``featurizer.counts`` over chunks in parallel processes (hashing is stateless)."""
    if n_jobs <= 1 or len(texts) <= chunk:
        return featurizer.counts(texts)
    from joblib import Parallel, delayed

    parts = Parallel(n_jobs=n_jobs)(
        delayed(featurizer.counts)(texts[i : i + chunk]) for i in range(0, len(texts), chunk)
    )
    return sp.vstack(parts, format="csr")


@dataclass
class FitResult:
    model: CpvModel
    timings: dict[str, float]
    n_train: int
    nnz_per_doc: float
    pruning: dict[str, Any] = field(default_factory=dict)


def fit(
    train: pd.DataFrame,
    cfg: TrainConfig,
    validation: pd.DataFrame | None = None,
    log: Any = print,
) -> FitResult:
    timings: dict[str, float] = {}
    if cfg.max_train_rows and len(train) > cfg.max_train_rows:
        train = train.sample(cfg.max_train_rows, random_state=cfg.seed)
    featurizer = Featurizer(cfg.features)

    t0 = time.perf_counter()
    counts = parallel_counts(
        featurizer, texts_of(train, cfg.features.max_description_chars), cfg.n_jobs
    )
    featurizer.fit_idf(counts)
    x = featurizer.tfidf(counts)
    del counts
    timings["featurize_train_s"] = time.perf_counter() - t0
    nnz_per_doc = x.nnz / max(x.shape[0], 1)
    log(f"features: {x.shape}, {nnz_per_doc:.0f} nnz/doc, {timings['featurize_train_s']:.0f}s")

    x_val = None
    if validation is not None:
        x_val = featurizer.tfidf(
            parallel_counts(
                featurizer, texts_of(validation, cfg.features.max_description_chars), cfg.n_jobs
            )
        )

    heads: dict[str, Head] = {}
    pruning: dict[str, Any] = {}
    for level in cfg.levels:
        y = train[LEVEL_COLUMN[level]].to_numpy()
        labels, counts_ = np.unique(y, return_counts=True)
        keep_labels = set(labels[counts_ >= cfg.min_class_count])
        mask = np.fromiter((v in keep_labels for v in y), bool, len(y))
        est = _make_estimator(cfg)
        t0 = time.perf_counter()
        est.fit(x[mask], y[mask])
        timings[f"fit_{level}_s"] = time.perf_counter() - t0
        head = _to_head(level, est, cfg.prune_threshold if cfg.classifier != "cnb" else 0.0)
        if x_val is not None and validation is not None:
            index = {c: i for i, c in enumerate(head.classes)}
            y_val = np.array([index.get(v, -1) for v in validation[LEVEL_COLUMN[level]]])
            if cfg.classifier != "cnb" and cfg.prune_grid:
                pruning[level] = select_pruning(head, x_val, y_val, cfg)
                log(f"{level}: pruning {pruning[level]}")
            head.temperature = fit_temperature(head.scores(x_val), y_val)
        heads[level] = head
        density = head.weights.nnz / (head.weights.shape[0] * head.weights.shape[1])
        log(
            f"{level}: {len(head.classes)} classes, fit {timings[f'fit_{level}_s']:.0f}s, "
            f"T={head.temperature:.3f}, weight density {density:.3f}"
        )
        del est

    metadata: dict[str, Any] = {"train_config": cfg.to_dict(), "pruning": pruning}
    model = CpvModel(featurizer=featurizer, heads=heads, metadata=metadata)
    return FitResult(
        model=model,
        timings=timings,
        n_train=x.shape[0],
        nnz_per_doc=nnz_per_doc,
        pruning=pruning,
    )


def select_pruning(
    head: Head, x_val: sp.csr_matrix, y_val: np.ndarray, cfg: TrainConfig
) -> dict[str, Any]:
    """Prune ``head`` in place with the largest threshold that keeps accuracy."""
    known = y_val >= 0

    def accuracy(weights: sp.csr_matrix) -> float:
        scores = np.asarray((x_val @ weights).todense()) + head.intercept
        return float(np.mean((scores.argmax(axis=1) == y_val) & known))

    base_weights = head.weights
    base_acc = accuracy(base_weights)
    chosen, chosen_acc, chosen_w = cfg.prune_threshold, base_acc, base_weights
    trials = []
    for t in sorted(cfg.prune_grid):
        w = base_weights.copy()
        w.data[np.abs(w.data) < t] = 0.0
        w.eliminate_zeros()
        acc = accuracy(w)
        trials.append({"threshold": t, "nnz": int(w.nnz), "val_top1": acc})
        if acc >= base_acc - cfg.prune_tolerance:
            chosen, chosen_acc, chosen_w = t, acc, w
    head.weights = chosen_w
    return {
        "unpruned_nnz": int(base_weights.nnz),
        "unpruned_val_top1": base_acc,
        "threshold": chosen,
        "nnz": int(chosen_w.nnz),
        "val_top1": chosen_acc,
        "trials": trials,
    }


def rank(
    model: CpvModel, texts: Sequence[str], level: str, k: int = 5, batch: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` labels and probabilities for each text."""
    head = model.head(level)
    classes = np.asarray(head.classes, dtype=object)
    k = min(k, len(classes))
    labels_out, probs_out = [], []
    for start in range(0, len(texts), batch):
        p = model.predict_proba(texts[start : start + batch], level)
        top = np.argsort(-p, axis=1, kind="stable")[:, :k]
        labels_out.append(classes[top])
        probs_out.append(np.take_along_axis(p, top, axis=1))
    return np.vstack(labels_out), np.vstack(probs_out)
