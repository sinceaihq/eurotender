"""Convert the official CPV 2008 Genericode list into the packaged label file.

Source: Publications Office of the European Union, eForms SDK
(https://github.com/OP-TED/eForms-SDK, codelists/cpv.gc), CC BY 4.0.

    python scripts/build_cpv_labels.py --sdk-version 1.15.1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

URL = "https://raw.githubusercontent.com/OP-TED/eForms-SDK/{version}/codelists/cpv.gc"
OUT = Path(__file__).resolve().parents[1] / "src/eurotender/resources/cpv_labels.json.gz"


def parse(xml_bytes: bytes) -> dict[str, dict[str, object]]:
    # Trusted input: a pinned file from the Publications Office repository.
    root = ET.fromstring(xml_bytes)  # noqa: S314
    codes: dict[str, dict[str, object]] = {}
    for row in root.iter("Row"):
        values = {
            v.get("ColumnRef"): (v.findtext("SimpleValue") or "").strip() for v in row.iter("Value")
        }
        code = values.pop("code")
        values.pop("Name", None)
        parent = values.pop("parentCode", "") or None
        labels = {k.removesuffix("_label"): v for k, v in values.items() if v}
        codes[code] = {"parent": parent, "labels": labels}
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-version", default="1.15.1")
    args = parser.parse_args()
    url = URL.format(version=args.sdk_version)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
        raw = resp.read()
    codes = parse(raw)
    payload = {
        "source": url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "license": "CC-BY-4.0",
        "attribution": "Common Procurement Vocabulary (CPV 2008), "
        "Publications Office of the European Union",
        "codes": dict(sorted(codes.items())),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    # mtime=0 keeps the archive byte-for-byte reproducible
    OUT.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    print(f"{len(codes)} codes -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
