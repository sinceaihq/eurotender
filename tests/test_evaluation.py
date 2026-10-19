from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from eurotender.evaluation import (
    LabelSimilarityBaseline,
    evaluate_model,
    majority_ranking,
    measure_performance,
)
from eurotender.model import CpvModel

from .conftest import toy_frame


def test_majority_ranking() -> None:
    r = majority_ranking(["a", "b", "b", "c", "b", "a"], n_rows=2, k=2)
    assert r.tolist() == [["b", "a"], ["b", "a"]]


def test_label_similarity_baseline_is_sensible() -> None:
    base = LabelSimilarityBaseline("division")
    assert len(base.classes) == 45
    ranked = base.rank(["Construction work for buildings", "Medical equipments"], k=3)
    assert ranked.shape == (2, 3)
    assert ranked[0, 0] == "45000000"
    assert "33000000" in ranked[1]


def test_evaluate_model_structure(tiny_model: CpvModel) -> None:
    df = toy_frame(1)
    df["near_duplicate_of_earlier_split"] = [i % 2 == 0 for i in range(len(df))]
    res = evaluate_model(tiny_model, df, "division", min_language_rows=1)
    assert res["overall"]["n"] == len(df)
    assert res["overall"]["top1_accuracy"] == 1.0
    assert res["novel_only"]["n"] == len(df) // 2
    assert set(res["per_language"]) == {"eng", "fin", "deu"}
    assert res["label_coverage"] == 1.0
    assert res["top_confusions"] == []
    assert 0.0 <= res["overall"]["ece"] <= 1.0
    assert res["confidence_buckets"][0]["coverage"] == 1.0


def test_evaluate_model_counts_unknown_labels_as_errors(tiny_model: CpvModel) -> None:
    df = pd.DataFrame(
        {
            "title": ["Road works"],
            "description": [""],
            "language": ["eng"],
            "cpv_division": ["90000000"],  # never seen in training
        }
    )
    res = evaluate_model(tiny_model, df, "division", min_language_rows=1)
    assert res["label_coverage"] == 0.0
    assert res["overall"]["top1_accuracy"] == 0.0


def test_measure_performance(tiny_model_dir: Path) -> None:
    perf = measure_performance(tiny_model_dir, ["Road works", "Software"] * 10)
    assert perf["load_seconds"] >= 0
    assert perf["single_text_latency_ms"]["p50"] > 0
    assert perf["batch_throughput_docs_per_s"] > 0
    assert set(perf["file_sizes_bytes"]) == {
        "config.json",
        "model.safetensors",
        "group.safetensors",
    }
    if sys.platform != "win32":
        assert 1 < perf["peak_rss_mb"] < 100_000
