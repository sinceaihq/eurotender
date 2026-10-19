"""Regenerate tests/hashing_golden.json (only when the feature config changes)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import sklearn

from eurotender.model import FeatureConfig, Featurizer

TEXT = "Kunnan IT-palvelut ja Straßenbau, CPV 72000000-5"


def main() -> None:
    cfg = FeatureConfig()
    counts = Featurizer(cfg).counts([TEXT])
    out = {
        "text": TEXT,
        "features": asdict(cfg),
        "scikit_learn": sklearn.__version__,
        "indices": sorted(counts.indices.tolist()),
    }
    path = Path(__file__).resolve().parents[1] / "tests/hashing_golden.json"
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
