# EuroTender: a reproducible, leakage-controlled baseline for multilingual CPV classification of EU tenders

*Since AI technical report, version 0.1.0 (October 2026)*

## Abstract

Public buyers in the European Union publish several hundred thousand
competition notices a year on Tenders Electronic Daily (TED), in 24 languages,
each tagged with a Common Procurement Vocabulary (CPV) code. CPV codes drive
how suppliers discover tenders, yet open, reproducible models and benchmarks
for assigning them are scarce: existing public models cover one language, do
not document their evaluation split, or are commercial. We release (i) a
dataset of <!-- n_rows:start -->824,201<!-- n_rows:end --> eForms contract notices from 2024-01 to 2026-09 with
a temporal train/validation/test split, one row per procedure and explicit
near-duplicate flags, (ii) a CPU-only classifier — hashed word and character
n-gram TF-IDF with linear one-vs-rest heads — that ranks CPV divisions and
groups with calibrated probabilities, and (iii) the full pipeline, from the TED
API harvest to the evaluation tables. <!-- abstract_results:start -->On 234,471 notices published after the training period, the model ranks the correct CPV division first for 82.1 % of notices and among the top five for 94.2 % (macro-F1 67.1); on notices that do not repeat earlier ones the figures are 80.9 % and 93.8 %. At the 3-digit group level top-1 accuracy is 65.0 %. A single prediction takes about 2 ms on one CPU core.<!-- abstract_results:end -->

## 1. Problem and users

A CPV code is a hierarchical 8-digit code (9,454 codes in CPV 2008) chosen by
the buyer. Suppliers, procurement platforms and analysts use it to filter
notices; a wrong or overly generic code makes a tender invisible to the right
suppliers, and analysing spend by sector depends on it. Two tasks benefit from
an automatic suggester:

1. **Coding assistance**: suggesting codes to the person writing a notice, or
   to a platform re-coding notices from other sources.
2. **Discovery and routing**: deciding which sector a notice belongs to when
   building alerts or analytics across countries and languages.

Intended users are SMEs and the software and data providers that serve them,
public-sector researchers, and developers of open-data applications. The
target is a model that is cheap to run (no GPU), easy to audit, and honest
about where it is weak.

## 2. Prior work

- Kayte and Schneider-Kamp (2019) combine an LSTM and an SVM on TED tenders
  for tender creation and main-CPV prediction (*A Mixed Neural Network and
  Support Vector Machine Model for Tender Creation in the European Union TED
  Database*, KMIS 2019, doi:10.5220/0008362701390145).
- Siciliani, Tanzi, Basile and Lops (2023) generate CPV codes for Italian
  tenders with sequence-to-sequence models (*Automatic Generation of Common
  Procurement Vocabulary Codes*, CLiC-it 2023); a related Italian IT5 model
  trained on 3.2M ANAC tenders is public as `basilepp19/cpv-it5`.
- `MKaan/multilingual-cpv-sector-classifier` fine-tunes multilingual BERT on
  TED for the 45 CPV divisions and reports F1 0.686; the split and data
  period are not documented.
- `mnavas/roberta-finetuned-CPV_Spanish` fine-tunes a Spanish RoBERTa on 2019
  Spanish procurement data for CPV divisions (reported F1 0.79, multi-label).
- Several commercial providers offer CPV classification and TED datasets;
  public Hugging Face TED datasets we found are small samples of paid products.

What is missing is an *open, multilingual, time-split* benchmark with
documented leakage controls and a strong, cheap baseline against which
heavier models can be compared. That is the gap EuroTender addresses. Because
evaluation protocols differ, the numbers above are **not** comparable with
ours and we do not claim to beat them.

## 3. Data

Full provenance is in [DATA.md](DATA.md); the essentials:

- **Source**: TED Search API v3, notice type `cn-standard` (eForms
  competition notices: subtypes 16, 17, 18, 19 and E3), published
  2024-01-01 … 2026-09-30, harvested in October 2026. Reuse is permitted by
  the TED legal notice; we release the processed data under CC BY 4.0.
- **Input**: the procedure title and description in the notice's first
  official language; **label**: the buyer-assigned main CPV code, evaluated at
  division (2-digit) and group (3-digit) level.
- **Privacy**: e-mail addresses, URLs, phone numbers and titled names are
  replaced by placeholders; buyer contact fields are never collected.
- **Leakage control**: one row per procedure (latest version); temporal split
  (train ≤ 2025-09-30 < validation ≤ 2025-12-31 < test); exact duplicates
  removed within each split; near-duplicates of earlier splits flagged and
  reported separately; anything that looks like an 8-digit CPV code is masked
  in the model input.

<!-- data_table:start -->
| split | published | notices | languages | divisions | near-duplicates of earlier splits |
|---|---|---|---|---|---|
| train | 2024-01-01 … 2025-09-30 | 510,016 | 23 | 45 | 0 (0.0 %) |
| validation | 2025-10-01 … 2025-12-31 | 79,714 | 24 | 45 | 9,639 (12.1 %) |
| test | 2026-01-01 … 2026-09-30 | 234,471 | 23 | 45 | 22,708 (9.7 %) |

Notices per language (all splits): German 207,824, French 147,577, Polish 90,276, Spanish 57,770, English 41,344, Italian 38,021, Dutch 31,012, Czech 28,244, Swedish 27,942, Romanian 26,041, Bulgarian 19,766, Portuguese 17,606, Lithuanian 14,337, Finnish 12,839, Greek 11,536, Latvian 9,212, Slovenian 8,494, Croatian 8,006, Hungarian 7,804, Danish 7,084, Slovak 6,347, Estonian 5,081, Irish 37, Maltese 1.

Data audit ([`results/data_audit.json`](../results/data_audit.json)): notice and
procedure identifiers are disjoint across splits and every date lies in its
window (hard assertions); 9,904 notices contain a CPV-like
code in their text and none survives normalisation; a broad e-mail pattern finds
0 residual matches after scrubbing. On a random sample of
3,000 test notices, 7.5 % have a training notice with
character-n-gram cosine similarity ≥ 0.9; the near-duplicate flag marks
10.2 % of the sample, and only
1.7 % of unflagged notices have such a neighbour.
<!-- data_table:end -->

## 4. Method

**Normalisation.** Unicode NFKC, contact-detail scrubbing, CPV-code masking,
Unicode case folding, whitespace collapsing.

**Features.** Two hashed blocks (scikit-learn `HashingVectorizer`,
MurmurHash3, 2^20 buckets each): word 1–2-grams and `char_wb` 2–5-grams.
Character n-grams matter in this setting: they share statistics across
inflected forms and compounds (Finnish, German, Hungarian, the Slavic
languages) and across closely related languages. Counts are transformed with
sublinear TF (log1p), a smoothed IDF fitted on the training split, and an L2
normalisation applied to each block separately, the char block receiving half
of the squared norm.

**Classifier.** One linear one-vs-rest head per CPV level, trained with
stochastic gradient descent. Classes with fewer than 10 training notices are
not predicted. After training, weights are magnitude-pruned with the largest
threshold whose validation top-1 accuracy stays within 0.1 points of the
unpruned head, and a softmax temperature is fitted on the validation split by
minimising negative log-likelihood.

**Why not a transformer?** We wanted a baseline that trains in minutes on a
CPU, runs in milliseconds, has no tokenizer or GPU dependency and can be
ported exactly to a browser. Section 6 compares it with a frozen multilingual
encoder under identical conditions.

**Model selection.** Pilot experiments on early-2024 data (train
Jan–Feb 2024, validation March 2024; not overlapping with the final
validation or test periods) compared Complement Naive Bayes, SGD with hinge,
modified-Huber and log losses, and LinearSVC, and ablated the character block
and description length. The final classifier and regularisation strength were
then chosen on the real validation split (`scripts/tune.py`). The test split
was used once, for the final evaluation.

<!-- pilot_table:start -->
**Pilot experiments** ([`results/pilot.json`](../results/pilot.json)):
pilot: train = notices published 2024-01-01..2024-02-29 (37,768), validation = 2024-03 (30,993); CPV division; built with the same pipeline before the email-obfuscation fix.

| round | classifier | regularisation | char share | max. description | top-1 | top-5 | macro-F1 |
|---|---|---|---|---|---|---|---|
| 1 | sgd_modified_huber | α=1e-05 | 0.5 | 2000 | 72.7 | 89.7 | 50.4 |
| 1 | sgd_modified_huber | α=2e-06 | 0.5 | 2000 | 69.3 | 85.4 | 43.0 |
| 1 | linear_svc (liblinear, max_iter=2000, single process) | C=0.5 | 0.5 | 2000 | 71.8 | 90.3 | 47.2 |
| 2 | cnb | α=0.1 (NB) | 0.5 | 2000 | 56.6 | 83.8 | 32.6 |
| 2 | sgd_modified_huber | α=3e-05 | 0.5 | 2000 | 72.6 | 90.2 | 49.2 |
| 2 | sgd_hinge | α=1e-05 | 0.5 | 2000 | 73.0 | 90.0 | 50.6 |
| 2 | sgd_log | α=1e-06 | 0.5 | 2000 | 71.5 | 89.7 | 46.5 |
| 2 | linear_svc (liblinear one-vs-rest in 6 processes, max_iter=200, tol=1e-3) | C=0.5 | 0.5 | 2000 | 71.8 | 90.3 | 47.2 |
| 2 | sgd_modified_huber | α=1e-05 | 0.0 | 2000 | 67.7 | 86.2 | 43.8 |
| 2 | sgd_modified_huber | α=1e-05 | 0.8 | 2000 | 72.5 | 89.5 | 50.6 |
| 2 | sgd_modified_huber | α=1e-05 | 0.5 | 500 | 72.5 | 89.5 | 50.2 |

Character n-grams are the most important design choice (+5.0 points top-1
over word features alone, same classifier). Truncating descriptions to 500
characters costs little, and the losses perform similarly; LinearSVC was not
better than SGD and is much slower to tune.
<!-- pilot_table:end -->

The two LinearSVC rows are separate runs (single process vs. one-vs-rest in
parallel) that converge to the same solution; only their wall-clock time
differs.

<!-- tuning_table:start -->
**Final selection on the validation split** ([`results/tuning.json`](../results/tuning.json);
fit on all 510,016 training notices, scored on 79,714
validation notices, CPV division, before pruning). The selection rule, fixed
before the runs, was the highest top-1 accuracy rounded to three decimals with
macro-F1 as tie-breaker; it chose **sgd_hinge, α = 1e-06** (bold).
The modified-Huber loss at α = 2e-6 is statistically indistinguishable and
slightly better at top-5; we kept the pre-registered choice.

| loss | α | fit time | top-1 | top-5 | macro-F1 | weighted-F1 |
|---|---|---|---|---|---|---|
| hinge | 5e-07 | 7.3 min | 82.7 | 94.7 | 68.6 | 82.2 |
| **hinge** | 1e-06 | 7.7 min | 82.9 | 94.6 | 68.6 | 82.4 |
| hinge | 2e-06 | 7.1 min | 82.2 | 94.2 | 66.9 | 81.6 |
| hinge | 5e-06 | 6.1 min | 80.5 | 93.2 | 63.6 | 79.5 |
| modified_huber | 5e-07 | 4.7 min | 82.2 | 94.7 | 68.0 | 81.7 |
| modified_huber | 1e-06 | 4.1 min | 82.5 | 95.0 | 68.5 | 82.1 |
| modified_huber | 2e-06 | 5.7 min | 82.8 | 95.3 | 68.8 | 82.3 |
| modified_huber | 5e-06 | 6.0 min | 82.4 | 95.6 | 67.3 | 81.7 |

Pruning (validation, tolerance 0.1 points top-1): division weights
48,468,694 → 11,627,113 (threshold 0.08),
group weights 127,223,669 → 33,287,176 (threshold 0.05).
<!-- tuning_table:end -->

## 5. Results

<!-- results:start -->
All numbers in this section come from `scripts/evaluate.py`
([`results/metrics.json`](../results/metrics.json), tables in
[`results/RESULTS.md`](../results/RESULTS.md)).

**Main results, test split.** "Novel only" removes notices flagged as exact
or near duplicates of an earlier split (9.7 % of the test set).

| level | classes | subset | n | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | ECE |
|---|---|---|---|---|---|---|---|---|---|
| division | 45 | all | 234,471 | 82.1 | 92.0 | 94.2 | 67.1 | 81.5 | 0.049 |
| division | 45 | novel only | 211,763 | 80.9 | 91.4 | 93.8 | 65.6 | 80.3 | 0.051 |
| group | 304 | all | 234,471 | 65.0 | 80.3 | 84.1 | 44.1 | 63.5 | 0.122 |
| group | 304 | novel only | 211,763 | 62.9 | 79.1 | 83.0 | 41.1 | 61.3 | 0.126 |

Validation-split division top-1 is 82.8 %; the test
figure is 0.8 points lower, consistent with
mild temporal drift over the nine-month test window.

**Baselines** (same test notices):

| system (CPV division, test) | top-1 | top-5 | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| majority class | 17.4 | 50.0 | 0.7 | 5.2 |
| CPV-label similarity (zero-shot) | 34.1 | 69.8 | 23.0 | 36.7 |
| Complement NB (same features) | 61.5 | 88.6 | 39.4 | 58.9 |
| **EuroTender** | **82.1** | **94.2** | **67.1** | **81.5** |

The zero-shot baseline matches notice text against the official CPV labels
in all 24 languages and uses no training notices; Complement Naive Bayes uses
the same features and training data as EuroTender.

**Memorisation vs. generalisation.** Accuracy on a random test sample,
grouped by the cosine similarity between the notice and its nearest training
notice:

| nearest-train cosine | n | top-1 | top-5 |
|---|---|---|---|
| 0.0–0.5 | 1,425 | 74.1 | 90.9 |
| 0.5–0.7 | 885 | 86.8 | 95.6 |
| 0.7–0.8 | 274 | 88.7 | 94.9 |
| 0.8–0.9 | 190 | 93.7 | 96.8 |
| 0.9–1.0 | 226 | 96.5 | 98.7 |

**Calibration.** Expected calibration error of the top-1 probability is
0.049 on the test split. Top-1 accuracy among notices whose top
probability exceeds a threshold:

| top probability | share of notices | top-1 accuracy |
|---|---|---|
| ≥ 0.0 | 100.0 % | 82.1 % |
| ≥ 0.5 | 83.4 % | 89.9 % |
| ≥ 0.7 | 75.3 % | 92.6 % |
| ≥ 0.9 | 63.3 % | 95.4 % |

**Per language** (CPV division, test; languages with < 50 test notices omitted):

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
<!-- results:end -->

## 6. Comparison with a multilingual encoder

<!-- embeddings:start -->
To check whether a neural encoder would be a better baseline, we trained both
approaches on the **same** random 100,000 training notices and scored
them on the same random 20,000 test notices (CPV division,
[`scripts/experiment_embeddings.py`](../scripts/experiment_embeddings.py),
[`results/embeddings.json`](../results/embeddings.json)). The encoder is
`intfloat/multilingual-e5-small` (revision `614241f622`, MIT licence), frozen, with
sentence embeddings (max. 256 tokens) fed to a multinomial
logistic regression (C = 10). Neither system was tuned for this subsample.

| system | top-1 | top-3 | top-5 | macro-F1 | test throughput (4 CPU threads) |
|---|---|---|---|---|---|
| EuroTender recipe (hashed TF-IDF + SGD) | 76.9 | 89.2 | 92.4 | 58.3 | 609 notices/s |
| multilingual-e5-small + logistic regression | 70.3 | 88.5 | 93.1 | 48.5 | 17 notices/s |

Encoding the 100,000 training notices took 101 minutes.
A *fine-tuned* encoder would likely do better than a frozen one; it needs a GPU
to train at this scale and is left for future work.
<!-- embeddings:end -->

## 7. Error analysis

<!-- errors:start -->
**Most frequent confusions** (test, CPV division):

| true division | predicted | notices | share of true class |
|---|---|---|---|
| 48 Software package and information systems | 72 IT services: consulting, software development, Internet and support | 1,322 | 23.7 % |
| 71 Architectural, construction, engineering and inspection services | 45 Construction work | 932 | 3.8 % |
| 72 IT services: consulting, software development, Internet and support | 48 Software package and information systems | 839 | 8.3 % |
| 79 Business services: law, marketing, consulting, recruitment, printing and security | 71 Architectural, construction, engineering and inspection services | 763 | 6.6 % |
| 45 Construction work | 71 Architectural, construction, engineering and inspection services | 694 | 1.7 % |
| 38 Laboratory, optical and precision equipments (excl. glasses) | 33 Medical equipments, pharmaceuticals and personal care products | 675 | 12.2 % |
| 79 Business services: law, marketing, consulting, recruitment, printing and security | 72 IT services: consulting, software development, Internet and support | 530 | 4.6 % |
| 72 IT services: consulting, software development, Internet and support | 79 Business services: law, marketing, consulting, recruitment, printing and security | 478 | 4.7 % |
| 90 Sewage, refuse, cleaning and environmental services | 71 Architectural, construction, engineering and inspection services | 432 | 2.6 % |
| 71 Architectural, construction, engineering and inspection services | 79 Business services: law, marketing, consulting, recruitment, printing and security | 410 | 1.7 % |

**Weakest divisions** (lowest test F1):

| division | support | precision | recall | F1 |
|---|---|---|---|---|
| 41 Collected and purified water | 40 | 50.0 | 10.0 | 16.7 |
| 51 Installation services (except software) | 682 | 49.5 | 13.6 | 21.4 |
| 75 Administration, defence and social security services | 982 | 58.4 | 19.6 | 29.3 |
| 76 Services related to the oil and gas industry | 94 | 64.5 | 21.3 | 32.0 |
| 65 Public utilities | 437 | 63.9 | 33.6 | 44.1 |
| 73 Research and development services and related consultancy services | 1,101 | 56.5 | 44.8 | 50.0 |
<!-- errors:end -->

Most errors fall into three groups.

1. **Overlapping CPV divisions.** CPV 48 (*software packages and information
   systems*) and 72 (*IT services*) describe the same purchase from two sides —
   a buyer procuring "a case-management system including implementation" can
   legitimately choose either — and the confusion runs in both directions.
   The same holds for 71 (*architectural and engineering services*) vs. 45
   (*construction work*) in design-and-build contracts, and 38 (*laboratory
   equipment*) vs. 33 (*medical equipment*) for hospital laboratories. These
   are disagreements with an ambiguous label as much as model errors; ranked
   suggestions (top-3/top-5) are the right interface for them.
2. **Rare or residual divisions.** Divisions such as 41 (*collected and
   purified water*), 76 (*oil and gas services*) and 65 (*public utilities*)
   have few training notices, and 51 (*installation services*) and 75
   (*administration services*) are often absorbed by the division of the
   equipment or service being installed or administered. Recall there is low
   and macro-F1 reflects it.
3. **Thin inputs.** Titles that only name a lot or a framework carry little
   signal; when the description is short as well, the model falls back on
   sector priors (see the model card for examples).

## 8. Efficiency

<!-- efficiency:start -->
Measured in a fresh process on AMD EPYC-Rome Processor
([`results/metrics.json`](../results/metrics.json), key `performance`):

- model files: `config.json` 0.0 MB, `model.safetensors` 109.8 MB, `group.safetensors` 274.7 MB
- load time 0.1 s; peak resident memory 748 MB
- single notice: p50 1.7 ms, p95 6.2 ms
- batch of 5,000: 661 notices/s (one process)
- training: 53 min for both heads (6 processes)

**Browser version.** The demo runs a JavaScript port of the pipeline whose
normalisation and hashed features are identical to the Python package
(checked feature by feature in `tests/test_web.py`). It loads a compact copy
with the 400,000 strongest features and int8 weights
(24 MB). On 20,000 test notices its top-1
prediction agrees with the full model for 96.1 % of notices;
its top-1 accuracy is 81.4 % vs. 81.8 % for the full model
([`results/web.json`](../results/web.json)).
<!-- efficiency:end -->

## 9. Limitations

- **Label noise.** CPV codes are chosen by buyers and are not consistent:
  the same kind of service is coded differently by different buyers. Part of
  the measured error is disagreement with a noisy label, not a model mistake,
  and no system can reach 100 %.
- **Granularity.** Only divisions (45) and groups are predicted; full codes
  are needed for some uses.
- **Inputs.** Only the procedure-level title and description are used; lot
  descriptions and tender documents are ignored.
- **Languages.** Performance differs by language (Section 5); the smallest
  languages have too few notices for reliable estimates.
- **Time.** The test period is January–September 2026; the model will drift
  as vocabulary and priorities change and should be re-trained periodically.
- **Privacy scrubbing** is rule-based and not a guarantee.

## 10. Safety and responsible use

Suggestions should be reviewed by a person. The model must not be used to
make legal determinations about whether a procurement is lawfully coded. It
is not designed to hold information about individuals: it stores hashed
n-gram weights, not text, and was trained on scrubbed notices, although a
personal name that slipped through scrubbing could in principle influence a
weight. The dataset contains public
notices published by public bodies; we removed contact details we could
detect and invite reports of anything we missed.

## 11. Future work

- Full 8-digit and multi-label prediction (all CPV codes of a notice).
- Using lot-level text and translations provided in multilingual notices.
- Fine-tuned multilingual encoders as a stronger (but more expensive)
  comparison point, evaluated on this exact split.
- A continuously updated benchmark with a rolling test window.

## 12. Reproduction

```bash
pip install -e ".[train]"
python -m eurotender.data.ted --start 2024-01 --end 2026-09 --out data/raw
python -m eurotender.data.build --raw data/raw --out data/processed
python scripts/audit_data.py --data data/processed --out results/data_audit.json
python scripts/tune.py --data data/processed --out results/tuning.json
python scripts/train.py --data data/processed --out models/eurotender-cpv-baseline \
    --config configs/baseline.json
python scripts/evaluate.py --data data/processed --model models/eurotender-cpv-baseline \
    --out results
python scripts/experiment_embeddings.py --data data/processed --out results/embeddings.json
python scripts/export_web.py --model models/eurotender-cpv-baseline --out web/model --keep 400000
python scripts/evaluate_web.py --model models/eurotender-cpv-baseline --web web/model \
    --data data/processed --rows 20000 --out results/web.json
```

`scripts/experiment_embeddings.py` additionally needs `torch` and
`sentence-transformers`; `scripts/evaluate_web.py` needs Node.js 18+.

The TED API returns the current state of the notice database, so a re-harvest
can differ slightly from ours (e.g. notices withdrawn later). The released
dataset and the per-month SHA-256 checksums in `stats.json` pin the exact
data used here.

## References

1. Publications Office of the European Union. *TED API documentation.* <https://docs.ted.europa.eu/api/latest/>
2. Publications Office of the European Union. *TED legal notice.* <https://ted.europa.eu/en/legal-notice>
3. Publications Office of the European Union. *eForms SDK 1.15.1.* <https://github.com/OP-TED/eForms-SDK>
4. Commission Regulation (EC) No 213/2008 amending Regulation (EC) No 2195/2002 on the Common Procurement Vocabulary (CPV).
5. S. Kayte, P. Schneider-Kamp. A Mixed Neural Network and Support Vector Machine Model for Tender Creation in the European Union TED Database. KMIS 2019. doi:10.5220/0008362701390145
6. L. Siciliani, E. Tanzi, P. Basile, P. Lops. Automatic Generation of Common Procurement Vocabulary Codes. CLiC-it 2023, pp. 396–402.
7. L. Wang et al. Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672, 2024.
8. F. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.
9. K. Weinberger et al. Feature Hashing for Large Scale Multitask Learning. ICML 2009.
10. C. Guo, G. Pleiss, Y. Sun, K. Weinberger. On Calibration of Modern Neural Networks. ICML 2017.
