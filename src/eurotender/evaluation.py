"""Held-out evaluation, baselines and performance measurements.

Requires the ``train`` extra (pandas, pyarrow).
"""

from __future__ import annotations

import gc
import os
import resource
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import cpv
from .metrics import classification_report, expected_calibration_error, per_class_f1
from .model import CpvModel
from .text import normalize
from .training import LEVEL_COLUMN, rank, texts_of


def majority_ranking(train_labels: Sequence[str], n_rows: int, k: int = 5) -> np.ndarray:
    """Rank labels by training frequency, identically for every row."""
    order = [label for label, _ in Counter(train_labels).most_common(k)]
    return np.tile(np.asarray(order, dtype=object), (n_rows, 1))


class LabelSimilarityBaseline:
    """Zero-shot baseline: nearest CPV label by character n-gram similarity.

    Every CPV code is represented by its labels in all 24 EU languages; a
    division scores the best similarity among its codes. No training notices
    are used.
    """

    def __init__(self, level: str = "division") -> None:
        table = cpv.codes()
        codes = sorted(table)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1
        )
        docs = [normalize(" ".join(table[c]["labels"].values())) for c in codes]
        self.matrix = self.vectorizer.fit_transform(docs)
        targets = [cpv.truncate(c, level) for c in codes]  # type: ignore[arg-type]
        self.classes = np.asarray(sorted(set(targets)), dtype=object)
        index = {c: i for i, c in enumerate(self.classes)}
        self._target = np.asarray([index[t] for t in targets])

    def rank(self, texts: Sequence[str], k: int = 5, batch: int = 2048) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch):
            x = self.vectorizer.transform([normalize(t) for t in texts[start : start + batch]])
            sims = np.asarray((x @ self.matrix.T).todense())
            best = np.full((sims.shape[0], len(self.classes)), -1.0)
            np.maximum.at(best.T, self._target, sims.T)
            out.append(self.classes[np.argsort(-best, axis=1, kind="stable")[:, :k]])
        return np.vstack(out)


def _report(y: np.ndarray, ranked: np.ndarray, conf: np.ndarray | None = None) -> dict[str, Any]:
    r = classification_report(y, ranked)
    if conf is not None:
        r["ece"] = expected_calibration_error(conf, ranked[:, 0] == y)
    return r


def evaluate_model(
    model: CpvModel,
    df: pd.DataFrame,
    level: str,
    k: int = 5,
    min_language_rows: int = 50,
) -> dict[str, Any]:
    texts = texts_of(df, model.featurizer.config.max_description_chars)
    y = df[LEVEL_COLUMN[level]].to_numpy(dtype=object)
    ranked, probs = rank(model, texts, level, k)
    conf = probs[:, 0]
    result: dict[str, Any] = {"overall": _report(y, ranked, conf)}
    if "near_duplicate_of_earlier_split" in df:
        novel = ~df.near_duplicate_of_earlier_split.to_numpy(dtype=bool)
        result["novel_only"] = _report(y[novel], ranked[novel], conf[novel])
        result["novel_share"] = float(novel.mean())
    known = set(model.head(level).classes)
    in_scope = np.fromiter((v in known for v in y), bool, len(y))
    result["label_coverage"] = float(in_scope.mean())

    langs = df.language.to_numpy()
    per_lang = {}
    for lang in sorted(set(langs)):
        m = langs == lang
        if m.sum() >= min_language_rows:
            per_lang[lang] = _report(y[m], ranked[m], conf[m])
    result["per_language"] = per_lang
    result["per_class"] = per_class_f1(list(y), list(ranked[:, 0]))

    pairs = Counter((t, p) for t, p in zip(y, ranked[:, 0], strict=True) if t != p).most_common(15)
    result["top_confusions"] = [
        {"true": t, "pred": p, "count": c, "share_of_true": c / int((y == t).sum())}
        for (t, p), c in pairs
    ]
    # Confidence buckets: how accurate are high-confidence suggestions?
    buckets = []
    for lo in (0.0, 0.5, 0.7, 0.9):
        m = conf >= lo
        buckets.append(
            {
                "min_confidence": lo,
                "coverage": float(m.mean()),
                "top1_accuracy": float((ranked[m, 0] == y[m]).mean()) if m.any() else None,
            }
        )
    result["confidence_buckets"] = buckets
    return result


def rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure_performance(
    model_dir: Path, sample_texts: Sequence[str], level: str = "division"
) -> dict[str, Any]:
    gc.collect()
    t0 = time.perf_counter()
    # What a default user loads: the main file only, further heads stay on disk.
    model = CpvModel.load(model_dir, levels=())
    load_s = time.perf_counter() - t0
    single = []
    for text in sample_texts[:300]:
        t = time.perf_counter()
        model.predict_proba([text], level)
        single.append((time.perf_counter() - t) * 1000)
    batch = list(sample_texts[:5000])
    t = time.perf_counter()
    model.predict_proba(batch, level)
    batch_s = time.perf_counter() - t
    files = {p.name: p.stat().st_size for p in model_dir.iterdir() if p.is_file()}
    return {
        "load_seconds": load_s,
        "single_text_latency_ms": {
            "p50": float(np.percentile(single, 50)),
            "p95": float(np.percentile(single, 95)),
            "mean": float(np.mean(single)),
        },
        "batch_throughput_docs_per_s": len(batch) / batch_s,
        "batch_size": len(batch),
        "peak_rss_mb": rss_mb(),
        "level": level,
        "heads_loaded": list(model.heads),
        "file_sizes_bytes": files,
        "cpu_count": os.cpu_count(),
    }
