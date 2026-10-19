---
language:
  - "bg"
  - "cs"
  - "da"
  - "de"
  - "el"
  - "en"
  - "es"
  - "et"
  - "fi"
  - "fr"
  - "hr"
  - "hu"
  - "it"
  - "lt"
  - "lv"
  - "nl"
  - "pl"
  - "pt"
  - "ro"
  - "sk"
  - "sl"
  - "sv"
license: "apache-2.0"
library_name: "eurotender"
pipeline_tag: "text-classification"
tags:
  - "procurement"
  - "public-procurement"
  - "cpv"
  - "ted"
  - "eu"
  - "multilingual"
  - "tf-idf"
  - "linear"
  - "cpu"
  - "safetensors"
datasets:
  - "sinceaihq/eurotender-cpv"
metrics:
  - "accuracy"
  - "f1"
model-index:
  - name: "eurotender-cpv-baseline"
    results:
      - task:
          type: "text-classification"
          name: "CPV division classification"
        dataset:
          type: "sinceaihq/eurotender-cpv"
          name: "EuroTender CPV (test split)"
          split: "test"
        metrics:
          - type: "accuracy"
            value: 0.8206
            name: "Top-1 accuracy"
          - type: "accuracy"
            value: 0.9416
            name: "Top-5 accuracy"
          - type: "f1"
            value: 0.6713
            name: "Macro F1"
          - type: "f1"
            value: 0.8154
            name: "Weighted F1"
      - task:
          type: "text-classification"
          name: "CPV group classification"
        dataset:
          type: "sinceaihq/eurotender-cpv"
          name: "EuroTender CPV (test split)"
          split: "test"
        metrics:
          - type: "accuracy"
            value: 0.6496
            name: "Top-1 accuracy"
          - type: "accuracy"
            value: 0.8406
            name: "Top-5 accuracy"
          - type: "f1"
            value: 0.4412
            name: "Macro F1"
          - type: "f1"
            value: 0.6351
            name: "Weighted F1"
---

# EuroTender CPV baseline

**Suggests Common Procurement Vocabulary (CPV) codes for European public
procurement notices, on a CPU. Accepts text in any EU language; evaluated on
22 of the 24 official languages.**

Given the title (and optionally the description) of a tender, the model ranks
CPV *divisions* (2-digit, 45 classes) and CPV *groups* (3-digit) with
calibrated probabilities. It is trained on 510,016
contract notices published on [TED](https://ted.europa.eu) between
2024-01-01 and 2025-09-30 and evaluated on
234,471 notices published **after** training data ends
(2026-01-01 … 2026-09-30).

Built by [Since AI](https://sinceai.ai) · code: [https://github.com/sinceaihq/eurotender](https://github.com/sinceaihq/eurotender) ·
data: [sinceaihq/eurotender-cpv](https://huggingface.co/datasets/sinceaihq/eurotender-cpv)

## Quick start

```bash
pip install "eurotender @ git+https://github.com/sinceaihq/eurotender@v0.1.0"
```

```python
from eurotender import predict_cpv

for p in predict_cpv("Development and maintenance of municipal IT systems", top_k=3):
    print(p.code, round(p.score, 3), p.label)
```

The package downloads `config.json` and `model.safetensors` (pinned to tag
`v0.1.0`, 110 MB) on first use; `group.safetensors`
(275 MB) only when a group prediction is first requested.
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
test notices that duplicate or nearly duplicate a notice from the training or
validation period (same text ignoring digits, or estimated character 5-gram
Jaccard similarity ≥ 0.8) — buyers often re-tender the same thing every year, so
"all" reflects real-world use and "novel only" is the harder, conservative number.

**CPV division (2-digit, 45 classes), test split**

| subset | n | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | ECE |
|---|---|---|---|---|---|---|---|
| all | 234,471 | 82.1 | 92.0 | 94.2 | 67.1 | 81.5 | 0.049 |
| novel only (90.3 % of test) | 211,763 | 80.9 | 91.4 | 93.8 | 65.6 | 80.3 | 0.051 |

**CPV group (3-digit, 304 classes), test split**

| subset | n | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 |
|---|---|---|---|---|---|---|
| all | 234,471 | 65.0 | 80.3 | 84.1 | 44.1 | 63.5 |
| novel only | 211,763 | 62.9 | 79.1 | 83.0 | 41.1 | 61.3 |

Group-level label coverage: 100.0 % of test notices have a
group seen at least 10 times in training (the rest count as errors).

**Baselines on the same test split (CPV division)**

| system | top-1 | top-5 | macro-F1 |
|---|---|---|---|
| majority class | 17.4 | 50.0 | 0.7 |
| CPV-label similarity (zero-shot) | 34.1 | 69.8 | 23.0 |
| Complement NB (same features) | 61.5 | 88.6 | 39.4 |
| **EuroTender (this model)** | **82.1** | **94.2** | **67.1** |

**Confidence is informative.** Top-1 accuracy among test notices whose top
probability is at least a threshold:

| top probability | share of notices | top-1 accuracy |
|---|---|---|
| ≥ 0.0 | 100.0 % | 82.1 % |
| ≥ 0.5 | 83.4 % | 89.9 % |
| ≥ 0.7 | 75.3 % | 92.6 % |
| ≥ 0.9 | 63.3 % | 95.4 % |

**Per language (CPV division, test split).** 22 languages have at least 50
test notices and are listed; Irish and Maltese have too few notices on TED to be evaluated.

| language | test notices | top-1 | top-3 | macro-F1 |
|---|---|---|---|---|
| German (`de`) | 58,254 | 87.6 | 95.1 | 65.5 |
| French (`fr`) | 39,932 | 79.8 | 90.9 | 61.5 |
| Polish (`pl`) | 27,618 | 87.9 | 95.3 | 74.7 |
| Spanish (`es`) | 16,340 | 79.0 | 90.7 | 65.0 |
| English (`en`) | 12,020 | 70.2 | 84.9 | 58.0 |
| Italian (`it`) | 11,194 | 80.9 | 91.4 | 63.1 |
| Dutch (`nl`) | 9,035 | 75.3 | 88.5 | 59.4 |
| Czech (`cs`) | 8,059 | 83.3 | 93.2 | 69.0 |
| Swedish (`sv`) | 7,850 | 71.9 | 86.0 | 55.1 |
| Romanian (`ro`) | 6,964 | 82.4 | 92.8 | 64.1 |
| Portuguese (`pt`) | 5,466 | 79.4 | 90.0 | 62.2 |
| Bulgarian (`bg`) | 5,069 | 87.2 | 94.9 | 65.8 |
| Lithuanian (`lt`) | 4,111 | 80.5 | 90.2 | 63.8 |
| Finnish (`fi`) | 3,688 | 77.2 | 89.8 | 61.5 |
| Greek (`el`) | 3,464 | 76.8 | 88.1 | 59.1 |
| Latvian (`lv`) | 2,920 | 81.3 | 91.0 | 61.9 |
| Slovenian (`sl`) | 2,533 | 79.9 | 90.3 | 64.7 |
| Croatian (`hr`) | 2,491 | 81.3 | 91.9 | 65.3 |
| Hungarian (`hu`) | 2,187 | 80.3 | 89.9 | 62.0 |
| Danish (`da`) | 1,980 | 78.2 | 88.3 | 52.9 |
| Slovak (`sk`) | 1,831 | 80.2 | 89.5 | 61.8 |
| Estonian (`et`) | 1,453 | 79.1 | 88.2 | 57.6 |

Strongest divisions: 66 Financial and insurance services (F1 0.93), 45 Construction work (F1 0.93), 33 Medical equipments, pharmaceuticals and personal care products (F1 0.92), 09 Petroleum products, fuel, electricity and other sources of energy (F1 0.91), 90 Sewage, refuse, cleaning and environmental services (F1 0.91).
Weakest divisions: 41 Collected and purified water (F1 0.17), 51 Installation services (except software) (F1 0.21), 75 Administration, defence and social security services (F1 0.29), 76 Services related to the oil and gas industry (F1 0.32), 65 Public utilities (F1 0.44).
The full per-division table and the most frequent confusions are in the
[evaluation report](https://github.com/sinceaihq/eurotender/blob/main/results/RESULTS.md).

## How it works

```
title + description (≤ 2000 chars)
  → NFKC, contact-detail scrubbing, CPV-code masking, casefold
  → hashed word (1, 2)-grams (2^20) + char_wb (2, 5)-grams (2^20)
  → sublinear TF-IDF, per-block L2 norm (char share 0.5)
  → linear one-vs-rest heads (sgd_hinge, alpha=1e-06) → softmax with fitted temperature
```

- Weights are stored as sparse CSR float32 matrices in `model.safetensors`
  (IDF + division head) and `group.safetensors` (group head); settings are in
  `config.json`. Loading never unpickles anything.
- Feature hashing means no vocabulary file; the hashing function is
  scikit-learn's MurmurHash3 and is pinned by a regression test.
- Temperatures (division T=0.282, group T=0.200) were fitted on the
  validation split (2025-10…12) by minimising negative log-likelihood.

## Training details

| | |
|---|---|
| Training notices | 510,016 (train split) |
| Calibration | validation split, 79,714 notices |
| Hyperparameters | classifier `sgd_hinge`, alpha `1e-06`, min. class count 10, weight pruning \|w\| < 0.08 (division) / 0.05 (group), chosen on validation |
| Seed | 20251019 |
| Training time | 53 min on AMD EPYC-Rome Processor (6 jobs) |
| Libraries | scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1 |
| Code commit | `b12fe728b14d9520bab56b5710425e52677d1b7f` |
| Dataset files (SHA-256) | `train.parquet` 62e4510e27d8…, `validation.parquet` 6d4b13f53e7f…, `test.parquet` a526a36072f0… |

## Inference cost (measured, one CPU process)

- model load 0.1 s; peak RSS 748 MB
- single notice: p50 1.7 ms, p95 6.2 ms
- batch throughput: 661 notices/s
- files: config.json 0.0 MB, model.safetensors 109.8 MB, group.safetensors 274.7 MB

## Limitations

- **Labels are buyer-assigned** and partly inconsistent between buyers, so
  part of the measured error is label noise rather than model error.
- **Division and group only.** Full 8-digit codes are not predicted. When a
  buyer gave only a division-level code (e.g. `79000000`), the group target is
  that same code, so the group head also contains such division-level classes.
- **Short or generic inputs can be confidently wrong.** Some get low
  probabilities ("Services" → 0.25), but others inherit a strong sector prior
  from the training data ("Lot 3" → 45 *Construction work* at 0.96, "Dostawa"
  ["delivery"] → 33 at 0.92). Calibration is measured on real notices, not on
  such fragments; give the model a substantive title or description.
- **Rare classes** are learned less well, which is why macro-F1 is well below
  accuracy. Groups with fewer than 10 training notices are not predicted.
- **Language skew.** Accuracy differs by language (table above); Irish and Maltese
  are effectively untested because TED has almost no notices in them.
- **Drift.** Procurement vocabulary changes; the model reflects 2024–2025
  notices.

## Data and licence

- Model and code: Apache-2.0.
- Training data: TED notices, © European Union, reused under the
  [TED legal notice](https://ted.europa.eu/en/legal-notice) ("can be freely
  reused, for commercial or non-commercial purposes"). Contact details were
  scrubbed before training; see the [dataset card](https://huggingface.co/datasets/sinceaihq/eurotender-cpv).
- CPV labels: eForms SDK 1.15.1, Publications Office of the EU, CC BY 4.0.

## Citation

```bibtex
@software{eurotender_2026,
  title  = {EuroTender: multilingual CPV classification for EU public procurement},
  author = {{Since AI}},
  year   = {2026},
  url    = {https://github.com/sinceaihq/eurotender},
  version = {0.1.0}
}
```
