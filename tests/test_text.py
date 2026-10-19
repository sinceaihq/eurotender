from __future__ import annotations

import pytest

from eurotender.text import compose, mask_cpv_codes, normalize, scrub_pii


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Contact anna.virtanen@kunta.fi today", "Contact [EMAIL] today"),
        ("Lisätiedot: hanna.virtanen(at)yritys.fi.", "Lisätiedot: [EMAIL]."),
        ("noora.harju-saarinen[at]tuomi.fi", "[EMAIL]"),
        ("mail{AT}example.org", "[EMAIL]"),
        ("rechnungseingang@ stadt.lueneburg", "[EMAIL]"),
        ("See https://example.eu/tender?id=1 now", "See [URL] now"),
        ("See www.hankinnat.fi", "See [URL]"),
        ("puh. 040 123 4567", "puh. [PHONE]"),
        ("Tel.: +49 (0)30 1234567", "Tel.: [PHONE]"),
        ("call +358 9 310 1691", "call [PHONE]"),
        ("Τηλ. 210 1234567", "Τηλ. [PHONE]"),
        ("Ansprechpartner: Frau Dr. Anna Müller.", "Ansprechpartner: [NAME]."),
        ("Contact Mme Dupont", "Contact [NAME]"),
    ],
)
def test_scrub_pii_masks_contact_details(raw: str, expected: str) -> None:
    assert scrub_pii(raw) == expected


@pytest.mark.parametrize(
    "keep",
    [
        "Wert 500.000,00 EUR",
        "Zeitraum 01.01.2026 - 31.12.2028",
        "1.100 m Rohrleitungen DN40",
        "Budget 2 500 000 EUR, 12 months",
        "reference 00123456",
        "Drainage works",
        "M. le Maire",
    ],
)
def test_scrub_pii_keeps_amounts_dates_and_references(keep: str) -> None:
    assert scrub_pii(keep) == keep


def test_scrub_pii_empty() -> None:
    assert scrub_pii("") == ""


def test_cpv_codes_are_masked() -> None:
    assert "45000000" not in normalize("CPV 45000000-7 Bauarbeiten")
    assert "72212000" not in normalize("code 72212000")
    assert mask_cpv_codes("45000000 - 7").strip() == "cpvcode"
    # Shorter numbers (system ids, quantities) are kept.
    assert "79140" in normalize("Systeme 79140, 38110")


def test_normalize_casefolds_and_collapses_whitespace() -> None:
    assert normalize("  Straße\n\tBAU  ") == "strasse bau"
    assert normalize("") == ""


def test_normalize_is_idempotent() -> None:
    once = normalize("Kunnan IT-palvelut, puh. 040 123 4567, CPV 72000000-5")
    assert normalize(once) == once


def test_compose() -> None:
    assert compose("Title", "Body") == "Title\nBody"
    assert compose("Title", "Title") == "Title"
    assert compose("Title", "") == "Title"
    assert compose("T", "x" * 5000, max_description_chars=10) == "T\n" + "x" * 10
