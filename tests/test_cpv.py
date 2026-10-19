from __future__ import annotations

import pytest

from eurotender import cpv


def test_code_list_is_complete() -> None:
    codes = cpv.codes()
    assert len(codes) == 9454
    divisions = [c for c in codes if c.endswith("000000")]
    assert len(divisions) == 45
    assert all(len(c) == 8 and c.isdigit() for c in codes)


@pytest.mark.parametrize(
    ("raw", "clean"),
    [("45000000-7", "45000000"), ("72", "72000000"), ("722", "72200000"), (" 3310 ", "33100000")],
)
def test_clean_code(raw: str, clean: str) -> None:
    assert cpv.clean_code(raw) == clean


@pytest.mark.parametrize("bad", ["", "abc", "123456789"])
def test_clean_code_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError, match="not a CPV code"):
        cpv.clean_code(bad)


def test_truncate() -> None:
    assert cpv.truncate("72212000", "division") == "72000000"
    assert cpv.truncate("72212000", "group") == "72200000"
    assert cpv.truncate("72212000", "class") == "72210000"


def test_labels_are_multilingual() -> None:
    assert (
        cpv.label("72000000")
        == "IT services: consulting, software development, Internet and support"
    )
    assert cpv.label("72000000", "fi") == cpv.label("72000000", "fin")
    assert cpv.label("72000000", "fi") != cpv.label("72000000", "en")
    assert cpv.label("72000000", "xx") == cpv.label("72000000", "en")  # falls back to English


def test_label_walks_up_to_known_ancestor() -> None:
    # 45110000 exists; 45119000 does not, so its label comes from 45110000.
    assert "45119000" not in cpv.codes()
    assert cpv.label("45119000") == cpv.label("45110000")


def test_is_valid() -> None:
    assert cpv.is_valid("45000000-7")
    assert not cpv.is_valid("99999999")
    assert not cpv.is_valid("not a code")
