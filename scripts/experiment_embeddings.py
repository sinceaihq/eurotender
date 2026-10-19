"""Controlled comparison: hashed TF-IDF + linear model vs. a multilingual encoder.

Both systems are trained on the same random subsample of the train split and
scored on the same random subsample of the test split, at CPV division level.
The encoder is frozen (sentence embeddings + multinomial logistic regression);
fine-tuning is out of scope for a CPU-only server.

Needs ``torch`` and ``sentence-transformers`` in addition to the train extra:

    python scripts/experiment_embeddings.py --data data/processed --out results/embeddings.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from eurotender.metrics import classification_report
from eurotender.model import FeatureConfig
from eurotender.training import TrainConfig, fit, load_split, rank, texts_of

ENCODER = "intfloat/multilingual-e5-small"
ENCODER_REVISION = "614241f622"  # pinned Hub commit (prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=100_000)
    parser.add_argument("--test-rows", type=int, default=20_000)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20251019)
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    cols = ["title", "description", "cpv_division"]
    train = load_split(args.data / "train.parquet", cols).sample(
        args.train_rows, random_state=args.seed
    )
    test = load_split(args.data / "test.parquet", cols).sample(
        args.test_rows, random_state=args.seed
    )
    y_train = train.cpv_division.to_numpy(dtype=object)
    y_test = test.cpv_division.to_numpy(dtype=object)
    results: dict[str, Any] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "seed": args.seed,
        "level": "division",
    }

    # 1) The EuroTender recipe on the same subsample.
    cfg = TrainConfig(
        levels=("division",), classifier="sgd_modified_huber", alpha=1e-5, n_jobs=args.threads
    )
    t0 = time.perf_counter()
    tfidf = fit(train, cfg, log=lambda *_: None).model
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ranked, _ = rank(tfidf, texts_of(test, FeatureConfig().max_description_chars), "division")
    infer_s = time.perf_counter() - t0
    results["tfidf_linear"] = {
        **classification_report(y_test, ranked),
        "fit_seconds": fit_s,
        "test_docs_per_second": len(test) / infer_s,
    }
    print(json.dumps(results["tfidf_linear"]), flush=True)

    # 2) Frozen multilingual encoder + logistic regression.
    encoder = SentenceTransformer(ENCODER, revision=ENCODER_REVISION, device="cpu")
    encoder.max_seq_length = 256

    def embed(texts: list[str]) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        # e5 models expect a task prefix; "query: " is the documented choice
        # for classification-style inputs.
        emb = encoder.encode(
            [f"query: {s}" for s in texts],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb), time.perf_counter() - t

    train_texts = [s[: args.max_chars] for s in texts_of(train, args.max_chars)]
    test_texts = [s[: args.max_chars] for s in texts_of(test, args.max_chars)]
    x_train, enc_train_s = embed(train_texts)
    x_test, enc_test_s = embed(test_texts)
    clf = LogisticRegression(C=10.0, max_iter=2000, random_state=args.seed)
    t0 = time.perf_counter()
    clf.fit(x_train, y_train)
    lr_s = time.perf_counter() - t0
    proba = clf.predict_proba(x_test)
    top = np.argsort(-proba, axis=1)[:, :5]
    ranked_e5 = np.asarray(clf.classes_, dtype=object)[top]
    results["e5_small_logreg"] = {
        **classification_report(y_test, ranked_e5),
        "encoder": ENCODER,
        "encoder_revision": ENCODER_REVISION,
        "max_seq_length": encoder.max_seq_length,
        "encode_train_seconds": enc_train_s,
        "fit_seconds": lr_s,
        "test_docs_per_second": len(test) / enc_test_s,
        "threads": args.threads,
    }
    print(json.dumps(results["e5_small_logreg"]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
