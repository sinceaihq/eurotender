"""Hyperparameter selection on the validation split (CPV division head).

Features are computed once; each grid point fits a classifier on the train
split and is scored on the validation split. The test split is not read.

    python scripts/tune.py --data data/processed --out results/tuning.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from eurotender.metrics import classification_report
from eurotender.model import Featurizer
from eurotender.training import (
    TrainConfig,
    _make_estimator,
    _to_head,
    load_split,
    parallel_counts,
    texts_of,
)

GRID: list[dict[str, Any]] = [
    {"classifier": loss, "alpha": alpha}
    for loss in ("sgd_hinge", "sgd_modified_huber")
    for alpha in (5e-7, 1e-6, 2e-6, 5e-6)
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()

    base = TrainConfig(n_jobs=args.n_jobs)
    cols = ["title", "description", "cpv_division"]
    train = load_split(args.data / "train.parquet", cols)
    val = load_split(args.data / "validation.parquet", cols)
    f = Featurizer(base.features)
    t0 = time.perf_counter()
    counts = parallel_counts(f, texts_of(train, base.features.max_description_chars), args.n_jobs)
    f.fit_idf(counts)
    x = f.tfidf(counts)
    del counts
    x_val = f.tfidf(
        parallel_counts(f, texts_of(val, base.features.max_description_chars), args.n_jobs)
    )
    feat_s = time.perf_counter() - t0
    print(f"features {x.shape} in {feat_s:.0f}s", flush=True)
    y, y_val = train.cpv_division.to_numpy(), val.cpv_division.to_numpy(dtype=object)

    runs = []
    for point in GRID:
        cfg = TrainConfig(**point, n_jobs=args.n_jobs)
        t0 = time.perf_counter()
        est = _make_estimator(cfg)
        est.fit(x, y)
        fit_s = time.perf_counter() - t0
        head = _to_head("division", est, prune_threshold=0.0)
        scores = head.scores(x_val)
        ranked = np.asarray(head.classes, dtype=object)[np.argsort(-scores, axis=1)[:, :5]]
        rep = classification_report(y_val, ranked)
        run = {**point, "fit_seconds": fit_s, **rep}
        runs.append(run)
        print(json.dumps(run), flush=True)
        del est, head

    best = max(runs, key=lambda r: (round(r["top1_accuracy"], 3), r["macro_f1"]))
    out = {
        "protocol": "fit on train, score on validation, CPV division, no pruning",
        "train_rows": int(x.shape[0]),
        "validation_rows": int(x_val.shape[0]),
        "featurize_seconds": feat_s,
        "runs": runs,
        "selected": {k: best[k] for k in ("classifier", "alpha")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("selected", out["selected"])


if __name__ == "__main__":
    main()
