from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from eurotender.data import ted


class FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}
        self.headers: dict[str, str] = {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    """Serves pages of notices and records the request bodies."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    def post(self, url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        assert url == ted.SEARCH_URL
        self.bodies.append(json)
        return self.responses.pop(0)

    def close(self) -> None:
        pass


def page(numbers: list[str], token: str | None, total: int) -> FakeResponse:
    notices = [{"publication-number": n, "links": {"xml": "x"}} for n in numbers]
    payload: dict[str, Any] = {"notices": notices, "totalNoticeCount": total}
    if token:
        payload["iterationNextToken"] = token
    return FakeResponse(200, payload)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ted.time, "sleep", lambda _s: None)


def test_build_query() -> None:
    q = ted.build_query(dt.date(2025, 1, 1), dt.date(2025, 2, 1), ["cn-standard"])
    assert (
        q == "notice-type=cn-standard AND publication-date>=20250101 AND publication-date<20250201"
    )
    q2 = ted.build_query(dt.date(2025, 1, 1), dt.date(2025, 2, 1), ["cn-standard", "cn-social"])
    assert q2.startswith("notice-type IN (cn-standard cn-social)")


@pytest.mark.parametrize(
    ("start", "end", "types"),
    [((2025, 2, 1), (2025, 1, 1), ["x"]), ((2025, 1, 1), (2025, 2, 1), [])],
)
def test_build_query_validates(
    start: tuple[int, int, int], end: tuple[int, int, int], types: list[str]
) -> None:
    with pytest.raises(ValueError, match=r"end must be after start|at least one notice type"):
        ted.build_query(dt.date(*start), dt.date(*end), types)


def test_month_range_crosses_year() -> None:
    months = ted.month_range("2024-11", "2025-02")
    assert [f"{m:%Y-%m}" for m in months] == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_iter_notices_follows_tokens_and_drops_links() -> None:
    session = FakeSession([page(["1", "2"], "t1", 3), page(["3"], "t2", 3), page([], None, 3)])
    out = list(ted.iter_notices("q", session=session))  # type: ignore[arg-type]
    assert [n["publication-number"] for n, _ in out] == ["1", "2", "3"]
    assert all("links" not in n for n, _ in out)
    assert "iterationNextToken" not in session.bodies[0]
    assert session.bodies[1]["iterationNextToken"] == "t1"
    assert all(b["paginationMode"] == "ITERATION" for b in session.bodies)


def test_iter_notices_retries_rate_limit() -> None:
    session = FakeSession([FakeResponse(429), FakeResponse(503), page(["1"], None, 1)])
    out = list(ted.iter_notices("q", session=session))  # type: ignore[arg-type]
    assert len(out) == 1
    assert len(session.bodies) == 3


def test_iter_notices_raises_on_client_error() -> None:
    session = FakeSession([FakeResponse(400, {"message": "bad query"})])
    with pytest.raises(ted.TedApiError, match="HTTP 400"):
        list(ted.iter_notices("q", session=session))  # type: ignore[arg-type]


def test_iter_notices_enforces_field_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        list(ted.iter_notices("q", fields=["f"] * 41, page_size=250))


def test_harvest_month_writes_manifest_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_iter(query: str, pause: float = 0.5) -> Any:
        calls.append(query)
        yield {"publication-number": "1-2025"}, 2
        yield {"publication-number": "1-2025"}, 2  # duplicate is ignored
        yield {"publication-number": "2-2025"}, 2

    monkeypatch.setattr(ted, "iter_notices", fake_iter)
    m = ted.harvest_month(dt.date(2025, 3, 1), tmp_path)
    assert m.harvested_count == 2
    assert m.total_notice_count == 2
    with gzip.open(tmp_path / "ted_2025-03.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert [json.loads(line)["publication-number"] for line in fh] == ["1-2025", "2-2025"]
    assert (
        json.loads((tmp_path / "ted_2025-03.manifest.json").read_text(encoding="utf-8"))["sha256"]
        == m.sha256
    )
    # Second call is a no-op thanks to the manifest.
    assert ted.harvest_month(dt.date(2025, 3, 1), tmp_path) == m
    assert len(calls) == 1


def test_harvest_month_detects_incomplete_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_iter(query: str, pause: float = 0.5) -> Any:
        yield {"publication-number": "1-2025"}, 5

    monkeypatch.setattr(ted, "iter_notices", fake_iter)
    with pytest.raises(ted.TedApiError, match="1 of 5"):
        ted.harvest_month(dt.date(2025, 3, 1), tmp_path)
    assert not list(tmp_path.glob("*.manifest.json"))
    assert not list(tmp_path.glob("*.tmp"))
