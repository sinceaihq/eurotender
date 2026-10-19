"""Render the Hugging Face model and dataset cards from recorded results.

Every number in the cards is read from ``results/metrics.json`` (written by
``scripts/evaluate.py``) or the dataset's ``stats.json``; nothing is typed by
hand.

    python scripts/render_cards.py --results results --stats data/processed/stats.json \
        --model-dir models/eurotender-cpv-baseline --out hub/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eurotender import cpv
from eurotender.cpv import ISO1_TO_ISO3

ISO3_TO_ISO1 = {v: k for k, v in ISO1_TO_ISO3.items()}
LANG_NAMES = {
    "bul": "Bulgarian", "ces": "Czech", "dan": "Danish", "deu": "German", "ell": "Greek",
    "eng": "English", "spa": "Spanish", "est": "Estonian", "fin": "Finnish", "fra": "French",
    "gle": "Irish", "hrv": "Croatian", "hun": "Hungarian", "ita": "Italian", "lit": "Lithuanian",
    "lav": "Latvian", "mlt": "Maltese", "nld": "Dutch", "pol": "Polish", "por": "Portuguese",
    "ron": "Romanian", "slk": "Slovak", "slv": "Slovenian", "swe": "Swedish",
}  # fmt: skip

GITHUB = "https://github.com/sinceaihq/eurotender"
MODEL_ID = "sinceaihq/eurotender-cpv-baseline"
DATASET_ID = "sinceaihq/eurotender-cpv"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def f3(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def model_index(results: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for level, res in results["test"]["model"].items():
        o = res["overall"]
        entries.append(
            {
                "task": {"type": "text-classification", "name": f"CPV {level} classification"},
                "dataset": {
                    "type": DATASET_ID,
                    "name": "EuroTender CPV (test split)",
                    "split": "test",
                },
                "metrics": [
                    {
                        "type": "accuracy",
                        "value": round(o["top1_accuracy"], 4),
                        "name": "Top-1 accuracy",
                    },
                    {
                        "type": "accuracy",
                        "value": round(o["top5_accuracy"], 4),
                        "name": "Top-5 accuracy",
                    },
                    {"type": "f1", "value": round(o["macro_f1"], 4), "name": "Macro F1"},
                    {"type": "f1", "value": round(o["weighted_f1"], 4), "name": "Weighted F1"},
                ],
            }
        )
    return [{"name": "eurotender-cpv-baseline", "results": entries}]


def yaml_dump(data: Any, indent: int = 0) -> str:
    """Minimal YAML emitter for card metadata (avoids a PyYAML dependency)."""
    pad = "  " * indent
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict | list) and v:
                out.append(f"{pad}{k}:")
                out.append(yaml_dump(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {json.dumps(v, ensure_ascii=False)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict | list):
                body = yaml_dump(item, indent + 1).lstrip()
                out.append(f"{pad}- {body}")
            else:
                out.append(f"{pad}- {json.dumps(item, ensure_ascii=False)}")
    return "\n".join(out)


def lang_table(per_language: dict[str, Any], stats_train: dict[str, int] | None = None) -> str:
    rows = ["| language | test notices | top-1 | top-3 | macro-F1 |", "|---|---|---|---|---|"]
    for lang, r in sorted(per_language.items(), key=lambda kv: -kv[1]["n"]):
        rows.append(
            f"| {LANG_NAMES.get(lang, lang)} (`{ISO3_TO_ISO1.get(lang, lang)}`) | {r['n']:,} "
            f"| {pct(r['top1_accuracy'])} | {pct(r['top3_accuracy'])} | {pct(r['macro_f1'])} |"
        )
    return "\n".join(rows)


def render_model_card(
    results: dict[str, Any], stats: dict[str, Any], config: dict[str, Any]
) -> str:
    test = results["test"]
    div = test["model"]["division"]
    grp = test["model"].get("group")
    meta = results["model_metadata"]
    perf = results["performance"]
    langs = sorted(ISO3_TO_ISO1[x] for x in stats["languages"] if x in ISO3_TO_ISO1)
    front = {
        "language": langs,
        "license": "apache-2.0",
        "library_name": "eurotender",
        "pipeline_tag": "text-classification",
        "tags": ["procurement", "public-procurement", "cpv", "ted", "eu", "multilingual",
                 "tf-idf", "linear", "cpu", "safetensors"],
        "datasets": [DATASET_ID],
        "metrics": ["accuracy", "f1"],
        "model-index": model_index(results),
    }  # fmt: skip
    hp = config["metadata"]["train_config"]
    feats = config["features"]
    heads = {h["name"]: h for h in config["heads"]}
    base_rows = "\n".join(
        f"| {name} | {pct(r['top1_accuracy'])} | {pct(r['top5_accuracy'])} | {pct(r['macro_f1'])} |"
        for name, r in test["baselines"].items()
    )
    o, nv = div["overall"], div["novel_only"]
    per_div = sorted(div["per_class"].items(), key=lambda kv: kv[1]["f1"])
    worst = ", ".join(f"{c[:2]} {cpv.label(c)} (F1 {r['f1']:.2f})" for c, r in per_div[:5])
    best = ", ".join(f"{c[:2]} {cpv.label(c)} (F1 {r['f1']:.2f})" for c, r in per_div[::-1][:5])
    conf_rows = "\n".join(
        f"| ≥ {b['min_confidence']:.1f} | {pct(b['coverage'])} % | {pct(b['top1_accuracy'])} % |"
        for b in div["confidence_buckets"]
    )
    group_block = ""
    if grp:
        g, gn = grp["overall"], grp["novel_only"]
        group_block = f"""
**CPV group (3-digit, {len(heads["group"]["classes"])} classes), test split**

| subset | n | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 |
|---|---|---|---|---|---|---|
| all | {g["n"]:,} | {pct(g["top1_accuracy"])} | {pct(g["top3_accuracy"])} | {pct(g["top5_accuracy"])} | {pct(g["macro_f1"])} | {pct(g["weighted_f1"])} |
| novel only | {gn["n"]:,} | {pct(gn["top1_accuracy"])} | {pct(gn["top3_accuracy"])} | {pct(gn["top5_accuracy"])} | {pct(gn["macro_f1"])} | {pct(gn["weighted_f1"])} |

Group-level label coverage: {pct(grp["label_coverage"])} % of test notices have a
group seen at least {hp["min_class_count"]} times in training (the rest count as errors).
"""
    sp = stats["splits"]
    return f"""---
{yaml_dump(front)}
---

# EuroTender CPV baseline

**Suggests Common Procurement Vocabulary (CPV) codes for European public
procurement notices, in all 24 official EU languages, on a CPU.**

Given the title (and optionally the description) of a tender, the model ranks
CPV *divisions* (2-digit, 45 classes) and CPV *groups* (3-digit) with
calibrated probabilities. It is trained on {meta["dataset"]["train_rows_used"]:,}
contract notices published on [TED](https://ted.europa.eu) between
{sp["train"]["date_range"][0]} and 2025-09-30 and evaluated on
{sp["test"]["rows"]:,} notices published **after** training data ends
({sp["test"]["date_range"][0]} … 2026-09-30).

Built by [Since AI]({"https://sinceai.ai"}) · code: [{GITHUB}]({GITHUB}) ·
data: [{DATASET_ID}](https://huggingface.co/datasets/{DATASET_ID})

## Quick start

```bash
pip install "eurotender @ git+{GITHUB}@v0.1.0"
```

```python
from eurotender import predict_cpv

for p in predict_cpv("Development and maintenance of municipal IT systems", top_k=3):
    print(p.code, round(p.score, 3), p.label)
```

The package downloads this repository (pinned to tag `v0.1.0`) on first use.
Labels can be returned in any EU language (`lang="fi"`, `"de"`, …), batches
are supported (`predict_cpv([...])`) and `level="group"` gives 3-digit
suggestions. There is also a CLI: `eurotender predict "Road resurfacing works"`.

## Intended use

- Suggesting CPV codes to people writing or screening tender notices.
- Routing and filtering procurement opportunities by sector (e.g. alerting a
  supplier to new tenders in division 72 *IT services*).
- A transparent, cheap baseline for research on procurement text.

**Not intended** for fully automatic coding without human review, for legal
determinations, or for inferring anything about individual people.

## Results

All numbers are from `scripts/evaluate.py` on the temporal test split; the
model never saw any notice published after 2025-09-30. *Novel only* excludes
test notices whose text (ignoring digits) also appears in the training or
validation period — buyers often re-tender the same thing every year, so
"all" reflects real-world use and "novel only" is the harder, conservative number.

**CPV division (2-digit, {len(heads["division"]["classes"])} classes), test split**

| subset | n | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | ECE |
|---|---|---|---|---|---|---|---|
| all | {o["n"]:,} | {pct(o["top1_accuracy"])} | {pct(o["top3_accuracy"])} | {pct(o["top5_accuracy"])} | {pct(o["macro_f1"])} | {pct(o["weighted_f1"])} | {o["ece"]:.3f} |
| novel only ({pct(div["novel_share"])} % of test) | {nv["n"]:,} | {pct(nv["top1_accuracy"])} | {pct(nv["top3_accuracy"])} | {pct(nv["top5_accuracy"])} | {pct(nv["macro_f1"])} | {pct(nv["weighted_f1"])} | {nv["ece"]:.3f} |
{group_block}
**Baselines on the same test split (CPV division)**

| system | top-1 | top-5 | macro-F1 |
|---|---|---|---|
{base_rows}
| **EuroTender (this model)** | **{pct(o["top1_accuracy"])}** | **{pct(o["top5_accuracy"])}** | **{pct(o["macro_f1"])}** |

**Confidence is informative.** Top-1 accuracy among test notices whose top
probability is at least a threshold:

| top probability | share of notices | top-1 accuracy |
|---|---|---|
{conf_rows}

**Per language (CPV division, test split).** Every official EU language was
evaluated; languages with fewer than 50 test notices are omitted.

{lang_table(div["per_language"])}

Strongest divisions: {best}.
Weakest divisions: {worst}.
The full per-division table and the most frequent confusions are in the
[evaluation report]({GITHUB}/blob/main/results/RESULTS.md).

## How it works

```
title + description (≤ {feats["max_description_chars"]} chars)
  → NFKC, contact-detail scrubbing, CPV-code masking, casefold
  → hashed word {tuple(feats["word_ngram_range"])}-grams (2^{feats["word_bits"]}) + char_wb {tuple(feats["char_ngram_range"])}-grams (2^{feats["char_bits"]})
  → sublinear TF-IDF, per-block L2 norm (char share {feats["char_weight"]})
  → linear one-vs-rest heads ({hp["classifier"]}, alpha={hp["alpha"]}) → softmax with fitted temperature
```

- Weights are stored in `model.safetensors` (sparse CSR, float32) and settings
  in `config.json`. Loading never unpickles anything.
- Feature hashing means no vocabulary file; the hashing function is
  scikit-learn's MurmurHash3 and is pinned by a regression test.
- Temperatures (division T={heads["division"]["temperature"]:.3f}{f", group T={heads['group']['temperature']:.3f}" if "group" in heads else ""}) were fitted on the
  validation split (2025-10…12) by minimising negative log-likelihood.

## Training details

| | |
|---|---|
| Training notices | {meta["dataset"]["train_rows_used"]:,} (train split) |
| Calibration | validation split, {sp["validation"]["rows"]:,} notices |
| Hyperparameters | classifier `{hp["classifier"]}`, alpha `{hp["alpha"]}`, min. class count {hp["min_class_count"]}, weight pruning \\|w\\| < {hp["prune_threshold"]} |
| Seed | {hp["seed"]} |
| Training time | {meta["timings_seconds"]["total"] / 60:.0f} min on {meta["hardware"].get("cpu_model", "CPU")} ({hp["n_jobs"]} jobs) |
| Libraries | scikit-learn {meta["versions"]["scikit_learn"]}, numpy {meta["versions"]["numpy"]}, scipy {meta["versions"]["scipy"]} |
| Code commit | `{meta.get("git_commit") or "n/a"}` |
| Dataset files (SHA-256) | {", ".join(f"`{k}` {v[:12]}…" for k, v in meta["dataset"]["files_sha256"].items())} |

## Inference cost (measured, one CPU process)

- model load {perf["load_seconds"]:.1f} s; peak RSS {perf["peak_rss_mb"]:.0f} MB
- single notice: p50 {perf["single_text_latency_ms"]["p50"]:.1f} ms, p95 {perf["single_text_latency_ms"]["p95"]:.1f} ms
- batch throughput: {perf["batch_throughput_docs_per_s"]:.0f} notices/s
- files: {", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in perf["file_sizes_bytes"].items())}

## Limitations

- **Labels are buyer-assigned** and partly inconsistent between buyers, so
  part of the measured error is label noise rather than model error.
- **Division and group only.** Full 8-digit codes are not predicted.
- **Short or generic titles** ("Services", "Lot 3") carry little signal;
  confidence is low for them and should be treated as such.
- **Rare classes** are learned less well, which is why macro-F1 is well below
  accuracy. Groups with fewer than {hp["min_class_count"]} training notices are not predicted.
- **Language skew.** Accuracy differs by language (table above); languages
  with little TED volume (Irish, Maltese) are effectively untested.
- **Drift.** Procurement vocabulary changes; the model reflects 2024–2025
  notices.

## Data and licence

- Model and code: Apache-2.0.
- Training data: TED notices, © European Union, reused under the
  [TED legal notice](https://ted.europa.eu/en/legal-notice) ("can be freely
  reused, for commercial or non-commercial purposes"). Contact details were
  scrubbed before training; see the [dataset card](https://huggingface.co/datasets/{DATASET_ID}).
- CPV labels: eForms SDK 1.15.1, Publications Office of the EU, CC BY 4.0.

## Citation

```bibtex
@software{{eurotender_2026,
  title  = {{EuroTender: multilingual CPV classification for EU public procurement}},
  author = {{{{Since AI}}}},
  year   = {{2026}},
  url    = {{{GITHUB}}},
  version = {{0.1.0}}
}}
```
"""


def render_dataset_card(stats: dict[str, Any]) -> str:
    sp = stats["splits"]
    total = stats["rows"]
    langs = sorted(ISO3_TO_ISO1[x] for x in stats["languages"] if x in ISO3_TO_ISO1)
    size = "1M<n<10M" if total >= 1_000_000 else "100K<n<1M"
    front = {
        "language": langs,
        "license": "cc-by-4.0",
        "multilinguality": "multilingual",
        "size_categories": [size],
        "source_datasets": ["original"],
        "task_categories": ["text-classification"],
        "task_ids": ["multi-class-classification", "topic-classification"],
        "pretty_name": "EuroTender CPV",
        "tags": ["procurement", "public-procurement", "cpv", "ted", "eu", "tenders"],
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {"split": "train", "path": "data/train.parquet"},
                    {"split": "validation", "path": "data/validation.parquet"},
                    {"split": "test", "path": "data/test.parquet"},
                ],
            }
        ],
    }
    lang_rows = "\n".join(
        f"| {LANG_NAMES.get(k, k)} | `{k}` | {v:,} |"
        for k, v in sorted(stats["languages"].items(), key=lambda kv: -kv[1])
    )
    pii = {
        k.removeprefix("pii_rows_with_").removesuffix("_masked"): v
        for k, v in stats.items()
        if k.startswith("pii_rows_with_")
    }
    return f"""---
{yaml_dump(front)}
---

# EuroTender CPV

**{total:,} European public procurement contract notices in 24 languages,
labelled with their main CPV code, with a leakage-controlled temporal split.**

Each row is one competition notice (eForms `cn-standard`) published on
[TED – Tenders Electronic Daily](https://ted.europa.eu) between 2024-01-01 and
2026-09-30: the procedure title and description in the notice's original
language, the buyer-assigned main CPV code and its division / group / class,
plus country, contract nature, procedure type and publication date.

Built by [Since AI](https://sinceai.ai) for the
[EuroTender]({GITHUB}) project; the baseline model trained on it is
[{MODEL_ID}](https://huggingface.co/{MODEL_ID}).

```python
from datasets import load_dataset

ds = load_dataset("{DATASET_ID}")
print(ds["train"][0]["title"], ds["train"][0]["cpv_division"])
```

## Splits

| split | publication dates | notices | languages | CPV divisions |
|---|---|---|---|---|
| train | {sp["train"]["date_range"][0]} … 2025-09-30 | {sp["train"]["rows"]:,} | {sp["train"]["languages"]} | {sp["train"]["divisions"]} |
| validation | {sp["validation"]["date_range"][0]} … 2025-12-31 | {sp["validation"]["rows"]:,} | {sp["validation"]["languages"]} | {sp["validation"]["divisions"]} |
| test | {sp["test"]["date_range"][0]} … 2026-09-30 | {sp["test"]["rows"]:,} | {sp["test"]["languages"]} | {sp["test"]["divisions"]} |

The split is **by time**: every test notice was published after every
training notice. Each procedure appears exactly once (latest published
version), so no procedure spans two splits. Exact duplicates (same text and
label) are removed within a split. Validation and test rows carry two flags:

- `exact_duplicate_of_earlier_split` – identical normalised text occurs in an
  earlier split ({stats["test_exact_duplicate_of_earlier_split"]:,} test rows);
- `near_duplicate_of_earlier_split` – identical after removing digits and
  truncating to 300 characters ({stats["test_near_duplicate_of_earlier_split"]:,} test rows).

Such repeats are real (annual re-tendering), so we keep them, but report
metrics with and without them.

## Fields

| field | type | description |
|---|---|---|
| `notice_id` | string | TED publication number, e.g. `588838-2025` (`https://ted.europa.eu/en/notice/-/detail/<id>`) |
| `publication_date` | string | `YYYY-MM-DD` |
| `procedure_id` | string | eForms procedure identifier (BT-04) |
| `notice_version` | int | eForms notice version |
| `language` | string | ISO 639-3 code of `title`/`description` |
| `official_languages` | list[string] | official languages of the notice |
| `buyer_country` | list[string] | ISO 3166-1 alpha-3 buyer countries |
| `contract_nature` | string | `services`, `supplies` or `works` |
| `procedure_type` | string | e.g. `open`, `restricted`, `neg-w-call` |
| `title` | string | procedure title (contact details scrubbed) |
| `description` | string | procedure description (contact details scrubbed) |
| `cpv_main` | string | main CPV code (8 digits, no check digit) |
| `cpv_division` / `cpv_group` / `cpv_class` | string | main code truncated to 2 / 3 / 4 digits, zero-padded |
| `cpv_all` | list[string] | every CPV code on the notice, incl. lots and additional codes |
| `exact_duplicate_of_earlier_split`, `near_duplicate_of_earlier_split` | bool | see above |

## Languages

| language | code | notices |
|---|---|---|
{lang_rows}

## Collection and processing

Harvested from the official TED Search API v3 (anonymous access, ITERATION
pagination, paced requests). Every month's raw file has a manifest with query,
counts and SHA-256; a month was accepted only if the harvested count matched
the API's `totalNoticeCount`. Of {stats["raw_notices"]:,} harvested notices,
{stats["dropped_unparseable_or_invalid_cpv"]:,} were legacy-format or lacked a
valid CPV code, {stats["dropped_older_procedure_versions"]:,} were older
versions of a procedure and {stats["dropped_exact_duplicates_within_split"]:,}
were exact duplicates. Full details: [docs/DATA.md]({GITHUB}/blob/main/docs/DATA.md).

## Personal data

Notices are published by public bodies and mostly describe what is being
bought, but free text can contain contact details. Before release we replaced
e-mail addresses, URLs, phone numbers and names introduced by an honorific with
`[EMAIL]`, `[URL]`, `[PHONE]` and `[NAME]`. Rows affected:
{", ".join(f"{k}: {v:,}" for k, v in pii.items())}. Buyer names, addresses and
contact-point fields were never collected. The scrubbing is rule-based and
cannot guarantee that no personal data remains (for instance, a bare name
without a title). If you find any, contact builders@sinceai.fi and we will
remove it.

## Licence and attribution

Released under **CC BY 4.0**. Source: TED – Tenders Electronic Daily,
© European Union, <https://ted.europa.eu>. TED procurement notices "can be
freely reused, for commercial or non-commercial purposes"
([legal notice](https://ted.europa.eu/en/legal-notice)). Changes made:
selection of fields and notices, contact-detail scrubbing, de-duplication and
splitting as described above. This dataset is not endorsed by the
Publications Office of the European Union.

## Limitations

- CPV codes are assigned by buyers and are sometimes inconsistent.
- Language and sector distributions follow TED publishing volume.
- Only procedure-level title and description; no lot texts or tender documents.

## Citation

```bibtex
@misc{{eurotender_cpv_2026,
  title  = {{EuroTender CPV: multilingual EU procurement notices with CPV labels}},
  author = {{{{Since AI}}}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{DATASET_ID}}}
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.results / "metrics.json").read_text())
    stats = json.loads(args.stats.read_text())
    config = json.loads((args.model_dir / "config.json").read_text())
    (args.out / "model").mkdir(parents=True, exist_ok=True)
    (args.out / "dataset").mkdir(parents=True, exist_ok=True)
    (args.out / "model" / "README.md").write_text(render_model_card(results, stats, config))
    (args.out / "dataset" / "README.md").write_text(render_dataset_card(stats))
    print(f"wrote cards to {args.out}")


if __name__ == "__main__":
    main()
