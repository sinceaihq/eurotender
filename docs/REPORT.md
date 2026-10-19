# EuroTender: a reproducible, leakage-controlled baseline for multilingual CPV classification of EU tenders

*Since AI technical report, version 0.1.0 (October 2026)*

## Abstract

Public buyers in the European Union publish several hundred thousand
competition notices a year on Tenders Electronic Daily (TED), in 24 languages,
each tagged with a Common Procurement Vocabulary (CPV) code. CPV codes drive
how suppliers discover tenders, yet open, reproducible models and benchmarks
for assigning them are scarce: existing public models cover one language, do
not document their evaluation split, or are commercial. We release (i) a
dataset of <!-- n_rows:start --><!-- n_rows:end --> eForms contract notices from 2024-01 to 2026-09 with
a temporal train/validation/test split, one row per procedure and explicit
near-duplicate flags, (ii) a CPU-only classifier — hashed word and character
n-gram TF-IDF with linear one-vs-rest heads — that ranks CPV divisions and
groups with calibrated probabilities, and (iii) the full pipeline, from the TED
API harvest to the evaluation tables. <!-- abstract_results:start --><!-- abstract_results:end -->

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

<!-- data_table:start --><!-- data_table:end -->

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

<!-- pilot_table:start --><!-- pilot_table:end -->

<!-- tuning_table:start --><!-- tuning_table:end -->

## 5. Results

<!-- results:start --><!-- results:end -->

## 6. Comparison with a multilingual encoder

<!-- embeddings:start --><!-- embeddings:end -->

## 7. Error analysis

<!-- errors:start --><!-- errors:end -->

## 8. Efficiency

<!-- efficiency:start --><!-- efficiency:end -->

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
make legal determinations about whether a procurement is lawfully coded, and
it contains no information about individuals. The dataset contains public
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
python scripts/tune.py --data data/processed --out results/tuning.json
python scripts/train.py --data data/processed --out models/eurotender-cpv-baseline \
    --config configs/baseline.json
python scripts/evaluate.py --data data/processed --model models/eurotender-cpv-baseline \
    --out results
python scripts/experiment_embeddings.py --data data/processed --out results/embeddings.json
```

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
