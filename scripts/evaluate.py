"""Evaluate a trained model and the baselines on held-out data.

    python scripts/evaluate.py --data ../data/eurotender/processed \
        --model models/eurotender-cpv-baseline --out results/
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurotender import cpv
from eurotender.evaluation import (
    LabelSimilarityBaseline,
    _report,
    evaluate_model,
    majority_ranking,
    measure_performance,
)
from eurotender.model import CpvModel
from eurotender.training import LEVEL_COLUMN, TrainConfig, fit, load_split, rank, texts_of

log = logging.getLogger("evaluate")


def fmt(x: float | None) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def markdown(results: dict[str, Any]) -> str:
    lines = ["# EuroTender evaluation results", ""]
    for split in ("validation", "test"):
        if split not in results:
            continue
        lines += [f"## {split.capitalize()} split", ""]
        for level, res in results[split]["model"].items():
            o, nv = res["overall"], res.get("novel_only", {})
            lines += [
                f"### Model, CPV {level} (n = {o['n']:,}, "
                f"label coverage {res['label_coverage']:.3f})",
                "",
                "| subset | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 "
                "| macro-P | macro-R | ECE |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for name, r in (("all", o), (f"novel only ({res.get('novel_share', 0):.1%})", nv)):
                if r:
                    lines.append(
                        f"| {name} | {fmt(r['top1_accuracy'])} | {fmt(r.get('top3_accuracy'))} "
                        f"| {fmt(r.get('top5_accuracy'))} | {fmt(r['macro_f1'])} "
                        f"| {fmt(r['weighted_f1'])} | {fmt(r['macro_precision'])} "
                        f"| {fmt(r['macro_recall'])} | {fmt(r.get('ece'))} |"
                    )
            lines.append("")
        base = results[split].get("baselines", {})
        if base:
            lines += [
                "### Baselines vs. model (CPV division)",
                "",
                "| system | top-1 | top-5 | macro-F1 | weighted-F1 |",
                "|---|---|---|---|---|",
            ]
            for name, r in base.items():
                lines.append(
                    f"| {name} | {fmt(r['top1_accuracy'])} | {fmt(r.get('top5_accuracy'))} "
                    f"| {fmt(r['macro_f1'])} | {fmt(r['weighted_f1'])} |"
                )
            m = results[split]["model"]["division"]["overall"]
            lines.append(
                f"| **EuroTender (this model)** | **{fmt(m['top1_accuracy'])}** "
                f"| **{fmt(m.get('top5_accuracy'))}** | **{fmt(m['macro_f1'])}** "
                f"| **{fmt(m['weighted_f1'])}** |"
            )
            lines.append("")
    test = results.get("test", {}).get("model", {}).get("division")
    if test:
        lines += [
            "## Per-language results (test, CPV division)",
            "",
            "| language | n | top-1 | top-3 | macro-F1 |",
            "|---|---|---|---|---|",
        ]
        for lang, r in sorted(test["per_language"].items(), key=lambda kv: -kv[1]["n"]):
            lines.append(
                f"| {lang} | {r['n']:,} | {fmt(r['top1_accuracy'])} | {fmt(r['top3_accuracy'])} "
                f"| {fmt(r['macro_f1'])} |"
            )
        lines += [
            "",
            "## Per-division results (test)",
            "",
            "| CPV | label | support | precision | recall | F1 |",
            "|---|---|---|---|---|---|",
        ]
        for code, r in sorted(test["per_class"].items(), key=lambda kv: -kv[1]["support"]):
            lines.append(
                f"| {code[:2]} | {cpv.label(code)} | {r['support']:,} | {fmt(r['precision'])} "
                f"| {fmt(r['recall'])} | {fmt(r['f1'])} |"
            )
        lines += [
            "",
            "## Most frequent confusions (test, CPV division)",
            "",
            "| true | predicted | count | share of true class |",
            "|---|---|---|---|",
        ]
        for c in test["top_confusions"]:
            lines.append(
                f"| {c['true'][:2]} {cpv.label(c['true'])} "
                f"| {c['pred'][:2]} {cpv.label(c['pred'])} "
                f"| {c['count']:,} | {c['share_of_true']:.1%} |"
            )
        lines += [
            "",
            "## Confidence vs. accuracy (test, CPV division)",
            "",
            "| min. confidence | coverage | top-1 |",
            "|---|---|---|",
        ]
        for b in test["confidence_buckets"]:
            lines.append(
                f"| {b['min_confidence']:.1f} | {b['coverage']:.3f} | {fmt(b['top1_accuracy'])} |"
            )
    sim = results.get("accuracy_by_train_similarity")
    if sim:
        lines += [
            "",
            "## Accuracy vs. similarity to the nearest training notice (test sample)",
            "",
            "| nearest-train cosine | n | top-1 | top-5 |",
            "|---|---|---|---|",
        ]
        for b in sim:
            lo, hi = b["cosine_range"]
            lines.append(
                f"| {lo:.1f}–{hi:.1f} | {b['n']:,} | {fmt(b['top1_accuracy'])} "
                f"| {fmt(b['top5_accuracy'])} |"
            )
    perf = results.get("performance")
    if perf:
        lines += [
            "",
            "## Inference performance (CPU)",
            "",
            f"- model load: {perf['load_seconds']:.2f} s",
            f"- single-text latency: p50 {perf['single_text_latency_ms']['p50']:.1f} ms, "
            f"p95 {perf['single_text_latency_ms']['p95']:.1f} ms",
            f"- batch throughput: {perf['batch_throughput_docs_per_s']:.0f} notices/s "
            f"(batch of {perf['batch_size']:,}, one process)",
            f"- peak RSS of a fresh process (heads loaded: "
            f"{', '.join(perf.get('heads_loaded', []))}; 5k texts): {perf['peak_rss_mb']:.0f} MB",
            "- files: "
            + ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in perf["file_sizes_bytes"].items()),
        ]
    return "\n".join(lines) + "\n"


def performance(model_dir: Path, data: Path) -> dict[str, Any]:
    max_desc = CpvModel.load(model_dir, levels=()).featurizer.config.max_description_chars
    texts = texts_of(load_split(data / "test.parquet").head(5000), max_desc)
    # Fresh process, so peak RSS reflects the model rather than this script.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        result: dict[str, Any] = pool.submit(measure_performance, model_dir, texts).result()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--skip-cnb", action="store_true", help="skip the Complement NB baseline")
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="re-measure inference performance and update an existing metrics.json",
    )
    args = parser.parse_args()
    if args.performance_only:
        results = json.loads((args.out / "metrics.json").read_text(encoding="utf-8"))
        results["performance"] = performance(args.model, args.data)
        results["model_metadata"] = CpvModel.load(args.model, levels=()).metadata
        (args.out / "metrics.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.out / "RESULTS.md").write_text(markdown(results), encoding="utf-8")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = CpvModel.load(args.model)
    max_desc = model.featurizer.config.max_description_chars
    train_labels = load_split(args.data / "train.parquet", ["cpv_division"]).cpv_division.tolist()
    label_baseline = LabelSimilarityBaseline("division")

    cnb = None
    if not args.skip_cnb:
        log.info("fitting Complement NB baseline on the train split")
        cols = ["title", "description", "cpv_division"]
        train = load_split(args.data / "train.parquet", cols)
        cfg = TrainConfig(
            features=model.featurizer.config,
            levels=("division",),
            classifier="cnb",
            min_class_count=1,
        )
        cnb = fit(train, cfg, log=log.info).model
        del train

    results: dict[str, Any] = {"model_metadata": model.metadata}
    for split in args.splits:
        df = load_split(args.data / f"{split}.parquet")
        log.info("%s: %d rows", split, len(df))
        res: dict[str, Any] = {"model": {}, "baselines": {}}
        for level in model.levels:
            res["model"][level] = evaluate_model(model, df, level)
            o = res["model"][level]["overall"]
            log.info(
                "%s %s: top1 %.4f macroF1 %.4f", split, level, o["top1_accuracy"], o["macro_f1"]
            )
        y = df[LEVEL_COLUMN["division"]].to_numpy(dtype=object)
        texts = texts_of(df, max_desc)
        res["baselines"]["majority class"] = _report(y, majority_ranking(train_labels, len(y)))
        res["baselines"]["CPV-label similarity (zero-shot)"] = _report(
            y, label_baseline.rank(texts)
        )
        if cnb is not None:
            ranked, _ = rank(cnb, texts, "division")
            res["baselines"]["Complement NB (same features)"] = _report(y, ranked)
        results[split] = res

    sim_path = args.out / "near_duplicate_sample.csv"
    if sim_path.exists():
        # Accuracy as a function of how similar a test notice is to its
        # nearest training notice (sample from scripts/audit_data.py).
        sims = pd.read_csv(sim_path, dtype={"notice_id": str})
        test = load_split(args.data / "test.parquet").merge(sims, on="notice_id")
        ranked, _ = rank(model, texts_of(test, max_desc), "division", 5)
        y = test.cpv_division.to_numpy(dtype=object)
        buckets = []
        for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
            m = ((test.nearest_train_cosine >= lo) & (test.nearest_train_cosine < hi)).to_numpy()
            buckets.append(
                {
                    "cosine_range": [lo, min(hi, 1.0)],
                    "n": int(m.sum()),
                    "top1_accuracy": float((ranked[m, 0] == y[m]).mean()) if m.any() else None,
                    "top5_accuracy": float(
                        np.mean([t in r for t, r in zip(y[m], ranked[m], strict=True)])
                    )
                    if m.any()
                    else None,
                }
            )
        results["accuracy_by_train_similarity"] = buckets

    results["performance"] = performance(args.model, args.data)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "RESULTS.md").write_text(markdown(results), encoding="utf-8")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
