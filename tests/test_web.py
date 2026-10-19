"""The browser port (web/eurotender.js) must agree with the Python package."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from eurotender.model import CpvModel, FeatureConfig, Featurizer
from eurotender.text import normalize

from .conftest import TOY

WEB = Path(__file__).resolve().parents[1] / "web"
ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

TEXTS = [
    *(t for t, _, _ in TOY),
    "Contact: anna.virtanen@kunta.fi, puh. 040 123 4567 or +358 9 310 1691",
    "Tel.: +49 (0)30 1234567, www.Example.de/Vergabe?x=1",
    "Yhteyshenkilö hanna.virtanen(at)yritys.fi, noora[AT]tuomi.fi, x @ y.fi",
    "Ansprechpartner: Frau Dr. Anna Müller, Herr Schmidt.",
    "ΟΔΟΣ ΑΘΗΝΩΝ — Straße, ﬁnance, Ǆ, İstanbul, ΣΊΣΥΦΟΣ",
    "CPV 45000000-7 Bauarbeiten; 79140, 38110, 1-7914-TK1",
    "Lot 12345678 and 00123456; 1.100 m Rohr DN40; 500.000,00 EUR",
    "Wert 500 000 EUR　Zeitraum 01.01.2026\t-\n31.12.2028",
    "x",
    "",
    "   leading and trailing   ",
    "naïve café — résumé — ½ — ①②③ — Ⅻ — x²",
    "Удостоверение — Ελλάδα — Mūsų — Łódź — Ærø — Þingvellir",
    "emoji 🚧🏗️ and CJK 東京 and Arabic عربى",
    "a_b c__d e-f 'g' h.i",
    "word " * 50,
]


def run_node(script: str, args: list[str], lines: list[str]) -> list[str]:
    proc = subprocess.run(
        ["node", str(WEB / script), *args],
        input="".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in lines),
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    return proc.stdout.splitlines()


@pytest.mark.parametrize("cfg", [FeatureConfig(), FeatureConfig(word_bits=12, char_bits=12)])
def test_js_features_match_python(cfg: FeatureConfig) -> None:
    out = run_node("check-features.mjs", [json.dumps(asdict(cfg))], TEXTS)
    assert len(out) == len(TEXTS)
    f = Featurizer(cfg)
    x = f.counts(TEXTS)
    x.sum_duplicates()
    for i, (text, line) in enumerate(zip(TEXTS, out, strict=True)):
        js = json.loads(line)
        assert js["norm"] == normalize(text), text
        row = x[i]
        py = sorted(zip(row.indices.tolist(), row.data.astype(int).tolist(), strict=True))
        assert [tuple(c) for c in js["counts"]] == py, text


def test_js_predictions_match_python(tiny_model: CpvModel, tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from export_web import export
    finally:
        sys.path.pop(0)
    # Keep every feature, so the only difference left is int8/float16
    # quantisation. The toy model is fitted to perfection, which drives its
    # fitted temperature towards zero and would amplify rounding errors
    # arbitrarily; T=1 isolates quantisation from calibration.
    model = copy.deepcopy(tiny_model)
    model.heads["division"].temperature = 1.0
    export(model, tmp_path, keep=10**9, levels=["division"])
    out = run_node("check-predict.mjs", [str(tmp_path)], TEXTS)
    head = model.heads["division"]
    proba = model.predict_proba(TEXTS, "division")
    for text, line, p in zip(TEXTS, out, proba, strict=True):
        web = json.loads(line)
        assert web[0][0] == head.classes[int(np.argmax(p))], text
        assert abs(web[0][1] - float(p.max())) < 0.02, text
