from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eurotender.data.build import build, near_key, parse_notice, text_key, write

SPLITS = {
    "train": ("2025-01-01", "2025-02-01"),
    "validation": ("2025-02-01", "2025-03-01"),
    "test": ("2025-03-01", "2025-04-01"),
}


def raw_notice(
    number: str,
    date: str,
    title: str,
    cpv: str = "72000000",
    proc: str | None = None,
    version: int = 1,
    desc: str = "",
    lang: str = "fin",
) -> dict[str, Any]:
    return {
        "publication-number": number,
        "publication-date": f"{date}+02:00",
        "notice-type": "cn-standard",
        "notice-version": version,
        "official-language": [lang.upper()],
        "buyer-country": ["FIN", "FIN"],
        "procedure-identifier": proc or f"proc-{number}",
        "procedure-type": "open",
        "contract-nature-main-proc": "services",
        "main-classification-proc": [cpv],
        "classification-cpv": [cpv, "72212000", cpv],
        "title-proc": {lang: title},
        "description-proc": {lang: desc},
    }


def write_raw(directory: Path, notices: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / "ted_2025-01.jsonl.gz", "wt") as fh:
        for n in notices:
            fh.write(json.dumps(n) + "\n")
    (directory / "ted_2025-01.manifest.json").write_text(
        json.dumps(
            {
                "month": "2025-01",
                "query": "q",
                "harvested_count": len(notices),
                "sha256": "x",
                "finished_at": "now",
            }
        )
    )


def test_parse_notice_basic() -> None:
    row = parse_notice(
        raw_notice("1-2025", "2025-01-05", "IT-palvelut", desc="Lisätiedot: matti@kunta.fi")
    )
    assert row is not None
    assert row["language"] == "fin"
    assert row["cpv_division"] == "72000000"
    assert row["cpv_group"] == "72000000"
    assert row["cpv_all"] == ["72000000", "72212000"]
    assert row["buyer_country"] == ["FIN"]
    assert row["publication_date"] == "2025-01-05"
    assert "matti@kunta.fi" not in row["description"]
    assert "[EMAIL]" in row["description"]


def test_parse_notice_prefers_official_language() -> None:
    raw = raw_notice("1-2025", "2025-01-05", "IT-palvelut")
    raw["official-language"] = ["SWE", "FIN"]
    raw["title-proc"] = {"fin": "IT-palvelut", "swe": "IT-tjänster"}
    row = parse_notice(raw)
    assert row is not None
    assert row["language"] == "swe"
    assert row["title"] == "IT-tjänster"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda n: n.update({"title-proc": {}}),  # legacy TED schema notice
        lambda n: n.update({"main-classification-proc": []}),
        lambda n: n.update({"main-classification-proc": ["99999999"]}),
        lambda n: n.update({"title-proc": {"fin": "  "}}),
    ],
)
def test_parse_notice_rejects_unusable(mutate: Any) -> None:
    n = raw_notice("1-2025", "2025-01-05", "IT-palvelut")
    mutate(n)
    assert parse_notice(n) is None


def test_duplicate_keys() -> None:
    assert text_key("A b", "") == text_key("a  B", "")
    assert near_key("Lot 1 cleaning", "") == near_key("Lot 2 cleaning", "")
    assert text_key("Lot 1 cleaning", "") != text_key("Lot 2 cleaning", "")


@pytest.fixture
def built(tmp_path: Path) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    notices = [
        raw_notice("1-2025", "2025-01-05", "Siivouspalvelut koululle", cpv="90910000"),
        # Same procedure republished later: only the latest version survives.
        raw_notice("2-2025", "2025-01-10", "IT-palvelut", proc="P1"),
        raw_notice("3-2025", "2025-02-10", "IT-palvelut (korjattu)", proc="P1", version=2),
        # Exact duplicate inside train.
        raw_notice("4-2025", "2025-01-06", "Siivouspalvelut koululle", cpv="90910000"),
        # Near duplicate of a train row in test (only the lot number differs).
        raw_notice("5-2025", "2025-01-07", "Tienrakennus osa 1", cpv="45233120"),
        raw_notice("6-2025", "2025-03-07", "Tienrakennus osa 2", cpv="45233120"),
        raw_notice("7-2025", "2025-03-08", "Lääkintälaitteet", cpv="33100000"),
        # Outside every split window.
        raw_notice("8-2025", "2025-06-01", "Myöhäinen", cpv="33100000"),
        # Legacy notice without eForms title.
        {**raw_notice("9-2025", "2025-01-02", "x"), "title-proc": {}},
    ]
    raw_dir = tmp_path / "raw"
    write_raw(raw_dir, notices)
    df, summary = build(raw_dir, SPLITS)
    return df, summary, raw_dir


def test_build_split_and_dedup(built: tuple[pd.DataFrame, dict[str, Any], Path]) -> None:
    df, summary, _ = built
    assert summary["raw_notices"] == 9
    assert summary["dropped_unparseable_or_invalid_cpv"] == 1
    assert summary["dropped_older_procedure_versions"] == 1
    assert summary["dropped_outside_split_windows"] == 1
    assert summary["dropped_exact_duplicates_within_split"] == 1
    assert set(df.notice_id) == {"1-2025", "3-2025", "5-2025", "6-2025", "7-2025"}
    split = dict(zip(df.notice_id, df.split, strict=True))
    assert split == {
        "1-2025": "train",
        "5-2025": "train",
        "3-2025": "validation",
        "6-2025": "test",
        "7-2025": "test",
    }


def test_build_has_no_cross_split_leakage(built: tuple[pd.DataFrame, dict[str, Any], Path]) -> None:
    df, _, _ = built
    assert df.groupby("procedure_id").split.nunique().max() == 1
    assert df.notice_id.is_unique
    for name, (lo, hi) in SPLITS.items():
        part = df[df.split == name]
        assert ((part.publication_date >= lo) & (part.publication_date < hi)).all()


def test_build_flags_near_duplicates(built: tuple[pd.DataFrame, dict[str, Any], Path]) -> None:
    df, summary, _ = built
    flags = dict(zip(df.notice_id, df.near_duplicate_of_earlier_split, strict=True))
    assert flags["6-2025"] is True or flags["6-2025"] == True  # noqa: E712
    assert not flags["7-2025"]
    assert summary["test_near_duplicate_of_earlier_split"] == 1
    assert summary["test_exact_duplicate_of_earlier_split"] == 0


def test_write_outputs(built: tuple[pd.DataFrame, dict[str, Any], Path], tmp_path: Path) -> None:
    df, summary, raw_dir = built
    out = tmp_path / "out"
    write(df, summary, out, raw_dir)
    for name in SPLITS:
        part = pd.read_parquet(out / f"{name}.parquet")
        assert "split" not in part.columns
        assert "text_hash" not in part.columns
    stats = json.loads((out / "stats.json").read_text())
    assert stats["raw_manifests"][0]["month"] == "2025-01"
    assert set(stats["files"]) == {"train.parquet", "validation.parquet", "test.parquet"}


def test_minhash_flags_reworded_duplicates() -> None:
    from eurotender.data import dedup

    base = (
        "Kunnossapitopalvelut kaupungin kiinteistöille vuosille, sisältäen lämmitys-, "
        "vesi- ja ilmanvaihtojärjestelmien huollon sekä päivystyksen ympäri vuorokauden."
    )
    reworded = base.replace("ympäri vuorokauden", "ympärivuorokautisesti")
    unrelated = "Lieferung von Büromöbeln für die Stadtverwaltung einschließlich Montage."
    sig_train = dedup.signatures([base])
    sig_test = dedup.signatures([reworded, unrelated])
    flags = dedup.flag_against_previous([sig_train, sig_test])
    assert flags[0].tolist() == [False]
    assert flags[1].tolist() == [True, False]


def test_minhash_signatures_parallel_equals_serial() -> None:
    from eurotender.data import dedup

    texts = [f"teksti numero {i} hankinta" for i in range(50)]
    assert (dedup.signatures(texts) == dedup.signatures(texts, n_jobs=2, chunk=20)).all()
