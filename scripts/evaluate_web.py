"""Measure how closely the browser copy of the model matches the full model.

    python scripts/evaluate_web.py --model models/eurotender-cpv-baseline --web web/model \
        --data data/processed --rows 20000 --out results/web.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from eurotender.model import CpvModel
from eurotender.training import load_split, rank, texts_of

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = CpvModel.load(args.model)
    df = load_split(args.data / f"{args.split}.parquet").sample(args.rows, random_state=7)
    texts = texts_of(df, model.featurizer.config.max_description_chars)
    full, full_p = rank(model, texts, "division", 5)
    payload = "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts)
    proc = subprocess.run(  # noqa: S603
        ["node", str(WEB_DIR / "check-predict.mjs"), str(args.web)],  # noqa: S607
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    web = [json.loads(line) for line in proc.stdout.splitlines()]
    web_top = np.array([[c for c, _ in row] for row in web], dtype=object)
    web_p = np.array([row[0][1] for row in web])
    y = df.cpv_division.to_numpy(dtype=object)
    size = sum(p.stat().st_size for p in args.web.iterdir() if p.name.startswith("web-model"))
    meta = json.loads((args.web / "web-model.json").read_text())
    result = {
        "split": args.split,
        "rows": len(texts),
        "web_model_bytes": size,
        "web_features_kept": {h["name"]: h["n_kept"] for h in meta["heads"]},
        "top1_agreement_with_full_model": float(np.mean(web_top[:, 0] == full[:, 0])),
        "top5_set_agreement": float(
            np.mean([set(a) == set(b) for a, b in zip(web_top, full, strict=True)])
        ),
        "mean_abs_top1_prob_diff": float(np.mean(np.abs(web_p - full_p[:, 0]))),
        "full_model_top1_accuracy": float(np.mean(full[:, 0] == y)),
        "web_model_top1_accuracy": float(np.mean(web_top[:, 0] == y)),
        "web_model_top5_accuracy": float(
            np.mean([t in row for t, row in zip(y, web_top, strict=True)])
        ),
        "node_timing": json.loads(proc.stderr.strip().splitlines()[-1]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
