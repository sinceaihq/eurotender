"""Audit the processed dataset for leakage and residual personal data.

Checks are hard assertions where a violation would invalidate the evaluation
(overlapping ids, dates outside their split) and measurements otherwise.

    python scripts/audit_data.py --data data/processed --out results/data_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from eurotender.text import CPV_CODE_RE, compose, normalize

SPLITS = ("train", "validation", "test")
WINDOWS = {
    "train": ("2024-01-01", "2025-10-01"),
    "validation": ("2025-10-01", "2026-01-01"),
    "test": ("2026-01-01", "2026-10-01"),
}
# Deliberately broader than the scrubber, to find what it missed.
EMAIL_LIKE = re.compile(r"[\w.+-]+\s?(?:@|\(at\)|\[at\])\s?[\w-]+\.[a-z]{2,}", re.IGNORECASE)
PHONE_LIKE = re.compile(r"(?:\+|00)\d{2,3}[\s./-]?\d{2,4}[\s./-]?\d{3,}")


def nearest_train_similarity(
    train_texts: list[str], query_texts: list[str], chunk: int = 500
) -> np.ndarray:
    """Max cosine similarity (char 3-5-gram TF-IDF) of each query to any train text."""
    hv = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), n_features=1 << 20, alternate_sign=False, norm=None
    )
    tf = TfidfTransformer(sublinear_tf=True)
    x_train = tf.fit_transform(hv.transform([normalize(t) for t in train_texts]))
    x_query = tf.transform(hv.transform([normalize(t) for t in query_texts]))
    x_train_t = x_train.T.tocsr()
    best = np.zeros(x_query.shape[0])
    for i in range(0, x_query.shape[0], chunk):
        sims = x_query[i : i + chunk] @ x_train_t
        best[i : i + chunk] = sims.max(axis=1).toarray().ravel()
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    args = parser.parse_args()

    dfs = {s: pd.read_parquet(args.data / f"{s}.parquet") for s in SPLITS}
    audit: dict[str, Any] = {"rows": {s: len(d) for s, d in dfs.items()}}

    for a in SPLITS:
        lo, hi = WINDOWS[a]
        d = dfs[a]
        assert ((d.publication_date >= lo) & (d.publication_date < hi)).all(), a
        assert d.notice_id.is_unique, a
        for b in SPLITS:
            if a < b:
                assert not set(d.notice_id) & set(dfs[b].notice_id), (a, b)
                assert not set(d.procedure_id) & set(dfs[b].procedure_id), (a, b)
    audit["ids_disjoint_across_splits"] = True
    audit["procedures_disjoint_across_splits"] = True
    audit["dates_within_split_windows"] = True

    allrows = pd.concat(dfs.values())
    texts = [compose(t, d) for t, d in zip(allrows.title, allrows.description, strict=True)]
    audit["rows_with_cpv_like_code_in_raw_text"] = int(
        sum(bool(CPV_CODE_RE.search(t)) for t in texts)
    )
    audit["rows_with_cpv_like_code_after_normalize"] = int(
        sum(bool(CPV_CODE_RE.search(normalize(t))) for t in texts)
    )
    assert audit["rows_with_cpv_like_code_after_normalize"] == 0
    audit["residual_email_like_rows"] = int(sum(bool(EMAIL_LIKE.search(t)) for t in texts))
    audit["residual_phone_like_rows"] = int(sum(bool(PHONE_LIKE.search(t)) for t in texts))
    examples = [t for t in texts if EMAIL_LIKE.search(t)][:5]
    audit["residual_email_like_examples_redacted"] = [
        EMAIL_LIKE.sub("<match>", e)[:160] for e in examples
    ]
    audit["empty_titles"] = int((allrows.title.str.strip() == "").sum())
    audit["label_distribution_train_top10"] = (
        dfs["train"].cpv_division.value_counts().head(10).to_dict()
    )
    audit["divisions_in_test_unseen_in_train"] = sorted(
        set(dfs["test"].cpv_division) - set(dfs["train"].cpv_division)
    )

    # Near-duplicate analysis on a random test sample against the full train split.
    sample = dfs["test"].sample(args.sample, random_state=11)
    train_texts = [
        compose(t, d) for t, d in zip(dfs["train"].title, dfs["train"].description, strict=True)
    ]
    query = [compose(t, d) for t, d in zip(sample.title, sample.description, strict=True)]
    sim = nearest_train_similarity(train_texts, query)
    audit["test_sample_size"] = args.sample
    audit["test_nearest_train_cosine"] = {
        "p50": float(np.percentile(sim, 50)),
        "p90": float(np.percentile(sim, 90)),
        "share_ge_0.9": float(np.mean(sim >= 0.9)),
        "share_ge_0.8": float(np.mean(sim >= 0.8)),
        "share_ge_0.95": float(np.mean(sim >= 0.95)),
    }
    flag = sample.near_duplicate_of_earlier_split.to_numpy()
    audit["heuristic_flag_vs_cosine"] = {
        "flagged_share": float(flag.mean()),
        "flagged_with_cosine_ge_0.9": float(np.mean(sim[flag] >= 0.9)) if flag.any() else None,
        "unflagged_with_cosine_ge_0.9": float(np.mean(sim[~flag] >= 0.9)),
    }
    sample_out = sample[["notice_id"]].copy()
    sample_out["nearest_train_cosine"] = sim
    sample_out.to_csv(args.out.with_name("near_duplicate_sample.csv"), index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
