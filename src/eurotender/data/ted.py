"""Harvest published contract notices from the TED Search API (v3).

The Search API is anonymous and documented at
https://docs.ted.europa.eu/api/latest/search.html. We use the ITERATION
(scroll) pagination mode, which is backed by an Elasticsearch point in time,
so a month is harvested as a consistent snapshot without duplicates.

Every month is written to its own gzipped JSONL file together with a small
manifest (query, fields, counts, checksum, timestamps). A month whose
manifest exists is skipped, which makes long harvests resumable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

SEARCH_URL = "https://api.ted.europa.eu/v3/notices/search"
USER_AGENT = "eurotender-harvester/0.1 (+https://github.com/sinceaihq/eurotender)"

# Fields requested for every notice. Keep (len(FIELDS) * page size) below the
# documented 10k fields-per-page limit.
FIELDS: tuple[str, ...] = (
    "publication-number",
    "publication-date",
    "notice-type",
    "notice-version",
    "official-language",
    "buyer-country",
    "procedure-identifier",
    "procedure-type",
    "contract-nature-main-proc",
    "main-classification-proc",
    "classification-cpv",
    "title-proc",
    "description-proc",
)
PAGE_SIZE = 250
MAX_FIELDS_PER_PAGE = 10_000

RETRY_STATUS = {429, 500, 502, 503, 504}


class TedApiError(RuntimeError):
    """Raised when the TED API returns a non-retryable error."""


@dataclass(frozen=True)
class MonthManifest:
    month: str
    query: str
    fields: list[str]
    api_url: str
    total_notice_count: int
    harvested_count: int
    file: str
    sha256: str
    started_at: str
    finished_at: str
    user_agent: str


def build_query(start: dt.date, end: dt.date, notice_types: Sequence[str]) -> str:
    """Expert-search query for notices published in ``[start, end)``."""
    if end <= start:
        raise ValueError("end must be after start")
    if not notice_types:
        raise ValueError("at least one notice type is required")
    if len(notice_types) == 1:
        types = f"notice-type={notice_types[0]}"
    else:
        types = f"notice-type IN ({' '.join(notice_types)})"
    return f"{types} AND publication-date>={start:%Y%m%d} AND publication-date<{end:%Y%m%d}"


def _post(
    session: requests.Session,
    body: dict[str, Any],
    *,
    max_retries: int = 8,
    timeout: float = 120.0,
) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(SEARCH_URL, json=body, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            log.warning("network error (%s), attempt %d/%d", exc, attempt, max_retries)
        else:
            if resp.status_code == 200:
                payload: dict[str, Any] = resp.json()
                if payload.get("timedOut"):
                    log.warning("search timed out server-side, attempt %d", attempt)
                else:
                    return payload
            elif resp.status_code in RETRY_STATUS:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                log.warning("HTTP %d, attempt %d/%d", resp.status_code, attempt, max_retries)
            else:
                raise TedApiError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        time.sleep(delay + random.uniform(0, 1))  # noqa: S311 - jitter, not crypto
        delay = min(delay * 2, 300.0)
    raise TedApiError(f"giving up after {max_retries} attempts")


def iter_notices(
    query: str,
    fields: Sequence[str] = FIELDS,
    *,
    page_size: int = PAGE_SIZE,
    pause: float = 0.5,
    session: requests.Session | None = None,
) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield ``(notice, total_notice_count)`` for every notice matching ``query``."""
    if page_size * len(fields) > MAX_FIELDS_PER_PAGE:
        raise ValueError("page_size * len(fields) exceeds the API limit")
    own_session = session is None
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", USER_AGENT)
    token: str | None = None
    try:
        while True:
            body: dict[str, Any] = {
                "query": query,
                "fields": list(fields),
                "limit": page_size,
                "scope": "ALL",
                "paginationMode": "ITERATION",
            }
            if token:
                body["iterationNextToken"] = token
            payload = _post(sess, body)
            notices = payload.get("notices") or []
            total = int(payload.get("totalNoticeCount") or 0)
            if not notices:
                return
            for notice in notices:
                notice.pop("links", None)  # derivable from publication-number
                yield notice, total
            token = payload.get("iterationNextToken")
            if not token:
                return
            time.sleep(pause)
    finally:
        if own_session:
            sess.close()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def month_range(start: str, end: str) -> list[dt.date]:
    """First days of every month from ``start`` to ``end`` inclusive (YYYY-MM)."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(dt.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _next_month(d: dt.date) -> dt.date:
    return dt.date(d.year + 1, 1, 1) if d.month == 12 else dt.date(d.year, d.month + 1, 1)


def harvest_month(
    month: dt.date,
    out_dir: Path,
    notice_types: Sequence[str] = ("cn-standard",),
    *,
    pause: float = 0.5,
) -> MonthManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ted_{month:%Y-%m}"
    data_path = out_dir / f"{stem}.jsonl.gz"
    manifest_path = out_dir / f"{stem}.manifest.json"
    if manifest_path.exists():
        log.info("%s already harvested, skipping", stem)
        return MonthManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))

    query = build_query(month, _next_month(month), notice_types)
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    tmp_path = data_path.with_suffix(".tmp")
    count, total = 0, 0
    seen: set[str] = set()
    with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
        for notice, total in iter_notices(query, pause=pause):
            pub = str(notice.get("publication-number", ""))
            if pub in seen:
                continue
            seen.add(pub)
            fh.write(json.dumps(notice, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            if count % 5000 == 0:
                log.info("%s: %d / %d", stem, count, total)
    if total and count < total:
        tmp_path.unlink()
        raise TedApiError(f"{stem}: harvested {count} of {total} notices")
    os.replace(tmp_path, data_path)
    manifest = MonthManifest(
        month=f"{month:%Y-%m}",
        query=query,
        fields=list(FIELDS),
        api_url=SEARCH_URL,
        total_notice_count=total,
        harvested_count=count,
        file=data_path.name,
        sha256=_sha256(data_path),
        started_at=started,
        finished_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        user_agent=USER_AGENT,
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    log.info("%s done: %d notices", stem, count)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True, help="first month, YYYY-MM")
    parser.add_argument("--end", required=True, help="last month (inclusive), YYYY-MM")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notice-type", action="append", default=None)
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between pages")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for month in month_range(args.start, args.end):
        harvest_month(month, args.out, args.notice_type or ("cn-standard",), pause=args.pause)


if __name__ == "__main__":
    main()
