"""Export a compact copy of a model for in-browser inference (web/eurotender.js).

The browser cannot reasonably download the full sparse weight matrices, so the
web copy keeps, per head, the ``--keep`` features with the largest absolute
weight, quantises their weights to int8 with one scale per class, and stores
the IDF vector as float16. ``scripts/evaluate_web.py`` measures how often the
web copy agrees with the full model.

Binary layout of ``web-model.bin`` (little endian):

    idf      float16[n_features]
    per head, at head.offset (4-byte aligned):
      ids    uint32[n_kept]                 sorted feature indices
      q      int8[n_kept * n_classes]       row-major, weight = q * scale[class]

    python scripts/export_web.py --model models/eurotender-cpv-baseline --out web/model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from eurotender import cpv
from eurotender.model import CpvModel


def export(model: CpvModel, out: Path, keep: int, levels: list[str]) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    assert model.featurizer.idf is not None
    blob = bytearray(model.featurizer.idf.astype("<f2").tobytes())
    heads = []
    for name in levels:
        head = model.head(name)
        w = sp.csc_matrix(head.weights.T)  # classes x features -> column access
        strength = np.asarray(abs(head.weights).max(axis=1).todense()).ravel()
        n_keep = min(keep, int((strength > 0).sum()))
        ids = np.sort(np.argpartition(-strength, n_keep - 1)[:n_keep]).astype("<u4")
        dense = np.asarray(w[:, ids].T.todense(), dtype=np.float32)  # n_keep x classes
        scale = np.abs(dense).max(axis=0) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(dense / scale), -127, 127).astype(np.int8)
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob += ids.tobytes() + q.tobytes()
        heads.append(
            {
                "name": name,
                "level": head.level,
                "classes": head.classes,
                "labels": {c: cpv.codes()[c]["labels"] for c in head.classes},
                "temperature": head.temperature,
                "intercept": [float(x) for x in head.intercept],
                "scale": [float(x) for x in scale],
                "n_kept": int(n_keep),
                "offset": offset,
            }
        )
    meta = {
        "format": "eurotender-web/1",
        "features": {
            **vars(model.featurizer.config),
            "word_ngram_range": list(model.featurizer.config.word_ngram_range),
            "char_ngram_range": list(model.featurizer.config.char_ngram_range),
        },
        "heads": heads,
        "source_model": model.metadata.get("name"),
        "source_git_commit": model.metadata.get("git_commit"),
    }
    (out / "web-model.bin").write_bytes(bytes(blob))
    (out / "web-model.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=200_000)
    parser.add_argument("--levels", nargs="+", default=["division"])
    args = parser.parse_args()
    model = CpvModel.load(args.model)
    export(model, args.out, args.keep, args.levels)
    size = (args.out / "web-model.bin").stat().st_size
    print(f"wrote {args.out} ({size / 1e6:.1f} MB binary)")


if __name__ == "__main__":
    main()
