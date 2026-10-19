"""Fill the generated sections of README.md and docs/REPORT.md from results/.

Sections are delimited by ``<!-- name:start -->`` / ``<!-- name:end -->``
markers and can be regenerated at any time; the prose around them is written
by hand, the numbers inside them are not.

    python scripts/render_docs.py --results results --stats data/processed/stats.json \
        --model-dir models/eurotender-cpv-baseline
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from eurotender import cpv, predict_cpv
from eurotender.model import CpvModel
from render_cards import ISO3_TO_ISO1, LANG_NAMES

ROOT = Path(__file__).resolve().parents[1]


def fill(text: str, name: str, content: str) -> str:
    pattern = re.compile(rf"<!-- {name}:start -->.*?<!-- {name}:end -->", re.DOTALL)
    if not pattern.search(text):
        raise KeyError(f"marker {name!r} not found")
    return pattern.sub(lambda _: f"<!-- {name}:start -->{content}<!-- {name}:end -->", text)


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def split_table(stats: dict[str, Any]) -> str:
    sp = stats["splits"]
    ends = {"train": "2025-09-30", "validation": "2025-12-31", "test": "2026-09-30"}
    rows = [
        "| split | published | notices | languages | divisions | near-duplicates of earlier splits |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("train", "validation", "test"):
        near = stats.get(f"{name}_near_duplicate_of_earlier_split", 0)
        share = near / sp[name]["rows"] if sp[name]["rows"] else 0
        rows.append(
            f"| {name} | {sp[name]['date_range'][0]} … {ends[name]} | {sp[name]['rows']:,} "
            f"| {sp[name]['languages']} | {sp[name]['divisions']} | {near:,} ({pct(share)} %) |"
        )
    return "\n".join(rows)


def level_table(model_res: dict[str, Any], heads: dict[str, Any]) -> str:
    rows = [
        "| level | classes | subset | n | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | ECE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for level, res in model_res.items():
        for name, r in (("all", res["overall"]), ("novel only", res["novel_only"])):
            rows.append(
                f"| {level} | {len(heads[level]['classes'])} | {name} | {r['n']:,} "
                f"| {pct(r['top1_accuracy'])} | {pct(r['top3_accuracy'])} | {pct(r['top5_accuracy'])} "
                f"| {pct(r['macro_f1'])} | {pct(r['weighted_f1'])} | {r['ece']:.3f} |"
            )
    return "\n".join(rows)


def baseline_table(test: dict[str, Any]) -> str:
    rows = [
        "| system (CPV division, test) | top-1 | top-5 | macro-F1 | weighted-F1 |",
        "|---|---|---|---|---|",
    ]
    for name, r in test["baselines"].items():
        rows.append(
            f"| {name} | {pct(r['top1_accuracy'])} | {pct(r['top5_accuracy'])} "
            f"| {pct(r['macro_f1'])} | {pct(r['weighted_f1'])} |"
        )
    o = test["model"]["division"]["overall"]
    rows.append(
        f"| **EuroTender** | **{pct(o['top1_accuracy'])}** | **{pct(o['top5_accuracy'])}** "
        f"| **{pct(o['macro_f1'])}** | **{pct(o['weighted_f1'])}** |"
    )
    return "\n".join(rows)


def language_table(per_language: dict[str, Any]) -> str:
    rows = ["| language | test notices | top-1 | top-3 | macro-F1 |", "|---|---|---|---|---|"]
    for lang, r in sorted(per_language.items(), key=lambda kv: -kv[1]["n"]):
        rows.append(
            f"| {LANG_NAMES.get(lang, lang)} (`{ISO3_TO_ISO1.get(lang, lang)}`) | {r['n']:,} "
            f"| {pct(r['top1_accuracy'])} | {pct(r['top3_accuracy'])} | {pct(r['macro_f1'])} |"
        )
    return "\n".join(rows)


def pilot_table(pilot: dict[str, Any]) -> str:
    rows = [
        "| round | classifier | regularisation | char share | max. description | top-1 | top-5 | macro-F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in pilot["runs"]:
        reg = (
            f"α={r['alpha']:g}"
            if r.get("alpha")
            else (f"C={r['c']:g}" if r.get("c") else "α=0.1 (NB)")
        )
        rows.append(
            f"| {r['round']} | {r['classifier']}{' (' + r['solver'] + ')' if r.get('solver') else ''} "
            f"| {reg} | {r['char_weight']} | {r['max_description_chars']} | {pct(r['top1_accuracy'])} "
            f"| {pct(r['top5_accuracy'])} | {pct(r['macro_f1'])} |"
        )
    return "\n".join(rows)


def tuning_table(tuning: dict[str, Any]) -> str:
    rows = [
        "| loss | α | fit time | top-1 | top-5 | macro-F1 | weighted-F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    sel = tuning["selected"]
    for r in tuning["runs"]:
        mark = "**" if r["classifier"] == sel["classifier"] and r["alpha"] == sel["alpha"] else ""
        rows.append(
            f"| {mark}{r['classifier'].removeprefix('sgd_')}{mark} | {r['alpha']:g} | {r['fit_seconds'] / 60:.1f} min "
            f"| {pct(r['top1_accuracy'])} | {pct(r['top5_accuracy'])} | {pct(r['macro_f1'])} "
            f"| {pct(r['weighted_f1'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    res = json.loads((args.results / "metrics.json").read_text())
    stats = json.loads(args.stats.read_text())
    tuning = json.loads((args.results / "tuning.json").read_text())
    pilot = json.loads((args.results / "pilot.json").read_text())
    audit = json.loads((args.results / "data_audit.json").read_text())
    emb_path = args.results / "embeddings.json"
    emb = json.loads(emb_path.read_text()) if emb_path.exists() else None
    web_path = args.results / "web.json"
    web = json.loads(web_path.read_text()) if web_path.exists() else None
    config = json.loads((args.model_dir / "config.json").read_text())
    heads = {h["name"]: h for h in config["heads"]}
    model = CpvModel.load(args.model_dir)

    test = res["test"]
    div, grp = test["model"]["division"], test["model"]["group"]
    o, nv, g = div["overall"], div["novel_only"], grp["overall"]
    pruning = config["metadata"]["pruning"]
    perf = res["performance"]

    # ------------------------------------------------------------------ README
    readme = (ROOT / "README.md").read_text()
    example = predict_cpv(
        "Development and maintenance of municipal IT systems", top_k=3, model=model
    )
    ex = "\n".join(f"{p.code} {round(p.score, 3)} {p.label}" for p in example)
    readme = fill(readme, "example", f"\n```text\n{ex}\n```\n")
    headline = f"""
Test split: {o["n"]:,} notices published January–September 2026, after the training period.

| CPV level | top-1 | top-3 | top-5 | macro-F1 |
|---|---|---|---|---|
| division (2-digit, {len(heads["division"]["classes"])} classes) | {pct(o["top1_accuracy"])} % | {pct(o["top3_accuracy"])} % | {pct(o["top5_accuracy"])} % | {pct(o["macro_f1"])} |
| division, novel notices only¹ | {pct(nv["top1_accuracy"])} % | {pct(nv["top3_accuracy"])} % | {pct(nv["top5_accuracy"])} % | {pct(nv["macro_f1"])} |
| group (3-digit, {len(heads["group"]["classes"])} classes) | {pct(g["top1_accuracy"])} % | {pct(g["top3_accuracy"])} % | {pct(g["top5_accuracy"])} % | {pct(g["macro_f1"])} |

¹ excluding the {pct(1 - div["novel_share"])} % of test notices that duplicate or nearly duplicate an earlier notice.

{baseline_table(test)}

Single notice on one CPU core: p50 {perf["single_text_latency_ms"]["p50"]:.0f} ms; batch throughput
{perf["batch_throughput_docs_per_s"]:,.0f} notices/s; model files {sum(perf["file_sizes_bytes"].values()) / 1e6:.0f} MB.
"""
    readme = fill(readme, "results", headline)
    (ROOT / "README.md").write_text(readme)

    # ------------------------------------------------------------------ REPORT
    rep = (ROOT / "docs/REPORT.md").read_text()
    rep = fill(rep, "n_rows", f"{stats['rows']:,}")
    rep = fill(
        rep,
        "abstract_results",
        f"On {o['n']:,} notices published after the training period, the model ranks "
        f"the correct CPV division first for {pct(o['top1_accuracy'])} % of notices and "
        f"among the top five for {pct(o['top5_accuracy'])} % (macro-F1 "
        f"{pct(o['macro_f1'])}); on notices that do not repeat earlier ones the figures "
        f"are {pct(nv['top1_accuracy'])} % and {pct(nv['top5_accuracy'])} %. At the "
        f"3-digit group level top-1 accuracy is {pct(g['top1_accuracy'])} %. A single "
        f"prediction takes about {perf['single_text_latency_ms']['p50']:.0f} ms on one CPU core.",
    )
    langs = sorted(stats["languages"].items(), key=lambda kv: -kv[1])
    lang_line = ", ".join(f"{LANG_NAMES.get(k, k)} {v:,}" for k, v in langs)
    rep = fill(
        rep,
        "data_table",
        f"""
{split_table(stats)}

Notices per language (all splits): {lang_line}.

Data audit ([`results/data_audit.json`](../results/data_audit.json)): notice and
procedure identifiers are disjoint across splits and every date lies in its
window (hard assertions); {audit["rows_with_cpv_like_code_in_raw_text"]:,} notices contain a CPV-like
code in their text and none survives normalisation; a broad e-mail pattern finds
{audit["residual_email_like_rows"]} residual matches after scrubbing. On a random sample of
{audit["test_sample_size"]:,} test notices, {pct(audit["test_nearest_train_cosine"]["share_ge_0.9"])} % have a training notice with
character-n-gram cosine similarity ≥ 0.9; the near-duplicate flag marks
{pct(audit["heuristic_flag_vs_cosine"]["flagged_share"])} % of the sample, and only
{pct(audit["heuristic_flag_vs_cosine"]["unflagged_with_cosine_ge_0.9"])} % of unflagged notices have such a neighbour.
""",
    )

    def pilot_run(**kw: Any) -> dict[str, Any]:
        return next(r for r in pilot["runs"] if all(r.get(k) == v for k, v in kw.items()))

    ref = {"classifier": "sgd_modified_huber", "alpha": 1e-5, "max_description_chars": 2000}
    char_gain = (
        pilot_run(**ref, char_weight=0.5)["top1_accuracy"]
        - pilot_run(**ref, char_weight=0.0)["top1_accuracy"]
    )
    rep = fill(
        rep,
        "pilot_table",
        f"""
**Pilot experiments** ([`results/pilot.json`](../results/pilot.json)):
{pilot["protocol"]}.

{pilot_table(pilot)}

Character n-grams are the most important design choice (+{pct(char_gain)} points top-1
over word features alone, same classifier). Truncating descriptions to 500
characters costs little, and the losses perform similarly; LinearSVC was not
better than SGD and is much slower to tune.
""",
    )
    rep = fill(
        rep,
        "tuning_table",
        f"""
**Final selection on the validation split** ([`results/tuning.json`](../results/tuning.json);
fit on all {tuning["train_rows"]:,} training notices, scored on {tuning["validation_rows"]:,}
validation notices, CPV division, before pruning). The selection rule, fixed
before the runs, was the highest top-1 accuracy rounded to three decimals with
macro-F1 as tie-breaker; it chose **{tuning["selected"]["classifier"]}, α = {tuning["selected"]["alpha"]:g}** (bold).
The modified-Huber loss at α = 2e-6 is statistically indistinguishable and
slightly better at top-5; we kept the pre-registered choice.

{tuning_table(tuning)}

Pruning (validation, tolerance 0.1 points top-1): division weights
{pruning["division"]["unpruned_nnz"]:,} → {pruning["division"]["nnz"]:,} (threshold {pruning["division"]["threshold"]}),
group weights {pruning["group"]["unpruned_nnz"]:,} → {pruning["group"]["nnz"]:,} (threshold {pruning["group"]["threshold"]}).
""",
    )
    sim = res.get("accuracy_by_train_similarity", [])
    sim_rows = "\n".join(
        f"| {b['cosine_range'][0]:.1f}–{b['cosine_range'][1]:.1f} | {b['n']:,} | {pct(b['top1_accuracy'])} | {pct(b['top5_accuracy'])} |"
        for b in sim
    )
    conf_rows = "\n".join(
        f"| ≥ {b['min_confidence']:.1f} | {pct(b['coverage'])} % | {pct(b['top1_accuracy'])} % |"
        for b in div["confidence_buckets"]
    )
    val = res["validation"]["model"]["division"]["overall"]
    rep = fill(
        rep,
        "results",
        f"""
All numbers in this section come from `scripts/evaluate.py`
([`results/metrics.json`](../results/metrics.json), tables in
[`results/RESULTS.md`](../results/RESULTS.md)).

**Main results, test split.** "Novel only" removes notices flagged as exact
or near duplicates of an earlier split ({pct(1 - div["novel_share"])} % of the test set).

{level_table(test["model"], heads)}

Validation-split division top-1 is {pct(val["top1_accuracy"])} %; the test
figure is {abs(100 * (val["top1_accuracy"] - o["top1_accuracy"])):.1f} points {"lower" if o["top1_accuracy"] < val["top1_accuracy"] else "higher"}, consistent with
mild temporal drift over the nine-month test window.

**Baselines** (same test notices):

{baseline_table(test)}

The zero-shot baseline matches notice text against the official CPV labels
in all 24 languages and uses no training notices; Complement Naive Bayes uses
the same features and training data as EuroTender.

**Memorisation vs. generalisation.** Accuracy on a random test sample,
grouped by the cosine similarity between the notice and its nearest training
notice:

| nearest-train cosine | n | top-1 | top-5 |
|---|---|---|---|
{sim_rows}

**Calibration.** Expected calibration error of the top-1 probability is
{o["ece"]:.3f} on the test split. Top-1 accuracy among notices whose top
probability exceeds a threshold:

| top probability | share of notices | top-1 accuracy |
|---|---|---|
{conf_rows}

**Per language** (CPV division, test; languages with < 50 test notices omitted):

{language_table(div["per_language"])}
""",
    )
    if emb:
        a, b = emb["tfidf_linear"], emb["e5_small_logreg"]
        rep = fill(
            rep,
            "embeddings",
            f"""
To check whether a neural encoder would be a better baseline, we trained both
approaches on the **same** random {emb["train_rows"]:,} training notices and scored
them on the same random {emb["test_rows"]:,} test notices (CPV division,
[`scripts/experiment_embeddings.py`](../scripts/experiment_embeddings.py),
[`results/embeddings.json`](../results/embeddings.json)). The encoder is
`{b["encoder"]}` (revision `{b["encoder_revision"]}`, MIT licence), frozen, with
sentence embeddings (max. {b["max_seq_length"]} tokens) fed to a multinomial
logistic regression (C = 10). Neither system was tuned for this subsample.

| system | top-1 | top-3 | top-5 | macro-F1 | test throughput ({b["threads"]} CPU threads) |
|---|---|---|---|---|---|
| EuroTender recipe (hashed TF-IDF + SGD) | {pct(a["top1_accuracy"])} | {pct(a["top3_accuracy"])} | {pct(a["top5_accuracy"])} | {pct(a["macro_f1"])} | {a["test_docs_per_second"]:,.0f} notices/s |
| multilingual-e5-small + logistic regression | {pct(b["top1_accuracy"])} | {pct(b["top3_accuracy"])} | {pct(b["top5_accuracy"])} | {pct(b["macro_f1"])} | {b["test_docs_per_second"]:,.0f} notices/s |

Encoding the {emb["train_rows"]:,} training notices took {b["encode_train_seconds"] / 60:.0f} minutes.
A *fine-tuned* encoder would likely do better than a frozen one; it needs a GPU
to train at this scale and is left for future work.
""",
        )
    worst = sorted(div["per_class"].items(), key=lambda kv: kv[1]["f1"])[:6]
    conf = div["top_confusions"][:10]
    rep = fill(
        rep,
        "errors",
        """
**Most frequent confusions** (test, CPV division):

| true division | predicted | notices | share of true class |
|---|---|---|---|
"""
        + "\n".join(
            f"| {c['true'][:2]} {cpv.label(c['true'])} | {c['pred'][:2]} {cpv.label(c['pred'])} | {c['count']:,} | {pct(c['share_of_true'])} % |"
            for c in conf
        )
        + """

**Weakest divisions** (lowest test F1):

| division | support | precision | recall | F1 |
|---|---|---|---|---|
"""
        + "\n".join(
            f"| {code[:2]} {cpv.label(code)} | {r['support']:,} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} |"
            for code, r in worst
        )
        + "\n",
    )
    eff = f"""
Measured in a fresh process on {config["metadata"]["hardware"].get("cpu_model", "the build server")}
([`results/metrics.json`](../results/metrics.json), key `performance`):

- model files: {", ".join(f"`{k}` {v / 1e6:.1f} MB" for k, v in perf["file_sizes_bytes"].items())}
- load time {perf["load_seconds"]:.1f} s; peak resident memory {perf["peak_rss_mb"]:.0f} MB
- single notice: p50 {perf["single_text_latency_ms"]["p50"]:.1f} ms, p95 {perf["single_text_latency_ms"]["p95"]:.1f} ms
- batch of {perf["batch_size"]:,}: {perf["batch_throughput_docs_per_s"]:,.0f} notices/s (one process)
- training: {config["metadata"]["timings_seconds"]["total"] / 60:.0f} min for both heads ({config["metadata"]["train_config"]["n_jobs"]} processes)
"""
    if web:
        eff += f"""
**Browser version.** The demo runs a JavaScript port of the pipeline whose
normalisation and hashed features are identical to the Python package
(checked feature by feature in `tests/test_web.py`). It loads a compact copy
with the {web["web_features_kept"]["division"]:,} strongest features and int8 weights
({web["web_model_bytes"] / 1e6:.0f} MB). On {web["rows"]:,} test notices its top-1
prediction agrees with the full model for {pct(web["top1_agreement_with_full_model"])} % of notices;
its top-1 accuracy is {pct(web["web_model_top1_accuracy"])} % vs. {pct(web["full_model_top1_accuracy"])} % for the full model
([`results/web.json`](../results/web.json)).
"""
    rep = fill(rep, "efficiency", eff)
    (ROOT / "docs/REPORT.md").write_text(rep)
    print("README.md and docs/REPORT.md updated")


if __name__ == "__main__":
    main()
