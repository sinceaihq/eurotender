# Data provenance and processing

This document describes exactly where the EuroTender data comes from, what
we do to it, and what we checked. The code that implements every step is in
[`src/eurotender/data/`](../src/eurotender/data).

## Source

| | |
|---|---|
| Publisher | Publications Office of the European Union, *Tenders Electronic Daily* (TED) |
| Interface | TED Search API v3, `POST https://api.ted.europa.eu/v3/notices/search` (anonymous) |
| Documentation | <https://docs.ted.europa.eu/api/latest/search.html> |
| Query | `notice-type=cn-standard AND publication-date>=YYYYMM01 AND publication-date<next month` |
| Pagination | `paginationMode=ITERATION` (Elasticsearch point in time; consistent per month) |
| Window | notices published 2024-01-01 … 2026-09-30 |
| Fields | `publication-number`, `publication-date`, `notice-type`, `notice-version`, `official-language`, `buyer-country`, `procedure-identifier`, `procedure-type`, `contract-nature-main-proc`, `main-classification-proc`, `classification-cpv`, `title-proc`, `description-proc` |

`cn-standard` is the eForms *competition notice* for the standard regime, i.e.
the notice a buyer publishes when it opens a competition — the notice type that
matters for opportunity discovery. In eForms SDK 1.15.1 it covers notice
subtypes 16 (general directive 2014/24/EU), 17 (sectoral directive
2014/25/EU), 18 (defence directive 2009/81/EC), 19 (concessions directive
2014/23/EU) and E3 (voluntary contract notice).

Each month is stored as a gzipped JSONL file plus a manifest recording the
query, field list, expected (`totalNoticeCount`) and harvested counts, SHA-256
of the file and timestamps. A month is only accepted if the harvested count
equals the count the API reports. Requests are paced (0.5 s between pages)
and retried with exponential back-off and jitter on HTTP 429/5xx.

## Reuse rights

From the [TED legal notice](https://ted.europa.eu/en/legal-notice):

> procurement notices … can be freely reused, for commercial or
> non-commercial purposes

System metadata is dedicated to the public domain (CC0 1.0), editorial content
is CC BY 4.0. The notice also warns that additional rights may need to be
cleared for content depicting identifiable private individuals, which is why
we scrub contact details (below) and do not redistribute buyer contact fields
at all. We do not use TED logos.

We redistribute the processed dataset under **CC BY 4.0** with the attribution
"Source: TED – Tenders Electronic Daily, © European Union,
https://ted.europa.eu", and state the changes we made (this document).

CPV labels come from the eForms SDK code list `codelists/cpv.gc`, release
1.15.1 (<https://github.com/OP-TED/eForms-SDK>, CC BY 4.0; SHA-256 of the
source file `e8f48fee…6301`).

## Processing

1. **Notice parsing.** Only eForms notices are used. Legacy TED-schema notices
   (still present in early 2024) have no `title-proc` and are skipped. The
   text is the procedure title and procedure description in the notice's
   first official language that has a title.
2. **Label.** `main-classification-proc`, the buyer-assigned main CPV code of
   the procedure, validated against CPV 2008. Derived columns hold the
   division (2 digits), group (3) and class (4).
3. **Contact-detail scrubbing** (`eurotender.text.scrub_pii`), applied to
   title and description: e-mail addresses (including the obfuscated forms
   `name(at)domain`, `name[at]domain`) → `[EMAIL]`, URLs → `[URL]`,
   international phone numbers and numbers following a telephone keyword in
   EU languages → `[PHONE]`, and names introduced by an honorific
   (Herr/Frau/Mr/Mrs/Ms/Mme/Dr/…) → `[NAME]`.
4. **One row per procedure.** TED republishes a procedure when it changes; we
   keep only the latest version (by publication date, then notice version).
   As a consequence no procedure can appear in two splits.
5. **Temporal split** by publication date: train 2024-01-01…2025-09-30,
   validation 2025-10-01…2025-12-31, test 2026-01-01…2026-09-30.
6. **Exact duplicates** (same normalised text and same main CPV code) are
   dropped within a split.
7. **Near-duplicate flags.** `exact_duplicate_of_earlier_split` marks
   validation/test rows whose normalised text occurs in an earlier split.
   `near_duplicate_of_earlier_split` additionally marks rows that match an
   earlier row after removing digits and truncating to 300 characters, **or**
   whose estimated Jaccard similarity of character 5-gram shingles with an
   earlier row is at least 0.8 (MinHash LSH, 64 permutations,
   [`data/dedup.py`](../src/eurotender/data/dedup.py)). These are legitimate in practice — buyers
   re-tender the same service every year — but we report metrics with and
   without them.
8. **Label masking at model input.** Buyers sometimes paste the CPV code into
   the description ("CPV 45000000-7"). The model's normaliser replaces every
   8-digit code with a placeholder so the classifier cannot read the answer.
   The dataset itself keeps the text as published (after scrubbing).

## Audit

`scripts/audit_data.py` re-checks the released files and writes
[`results/data_audit.json`](../results/data_audit.json): notice and procedure
identifiers are disjoint across splits, every publication date lies inside
its split window, no CPV-like code survives normalisation, and broad
e-mail / phone patterns are counted to find what the scrubber missed. It also
measures, for a random test sample, the TF-IDF cosine similarity to the
nearest training notice, to check how well the near-duplicate flag works.

## Known limitations of the data

- **Labels are buyer-assigned.** The main CPV code is chosen by the
  contracting authority and is sometimes inconsistent (e.g. a framework for
  IT consulting coded as 79 *Business services* by one buyer and 72 *IT
  services* by another). Measured accuracy therefore has a ceiling below 100 %.
- **Language imbalance** follows publishing volume: German, French and Polish
  dominate; Irish and Maltese are rare.
- **Scrubbing is rule-based** and cannot catch every personal identifier
  (e.g. a bare name without an honorific). Report anything you find to
  builders@sinceai.fi and we will remove it.
- Only the procedure-level title/description is used; lot-level texts and
  attached tender documents are not.
