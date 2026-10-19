"""Train a EuroTender model and save it with full provenance.

    python scripts/train.py --data ../data/eurotender/processed \
        --out models/eurotender-cpv-baseline --config configs/baseline.json

Protocol: the model is fitted on the train split only; the validation split
is used for softmax temperature calibration. The test split is never read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import platform
import subprocess
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

import eurotender
from eurotender.model import FeatureConfig
from eurotender.training import TrainConfig, fit, load_split

log = logging.getLogger("train")


def load_config(path: Path | None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    raw: dict[str, Any] = json.loads(path.read_text())
    feats = FeatureConfig.from_dict({**_defaults(FeatureConfig), **raw.pop("features", {})})
    if "levels" in raw:
        raw["levels"] = tuple(raw["levels"])
    return TrainConfig(features=feats, **raw)


def _defaults(cls: type) -> dict[str, Any]:
    inst = cls()
    return {f.name: getattr(inst, f.name) for f in fields(inst)}


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "-C", str(repo), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "os": platform.platform(),
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        model = next(line for line in cpuinfo.splitlines() if line.startswith("model name"))
        info["cpu_model"] = model.split(":", 1)[1].strip()
        info["logical_cpus"] = cpuinfo.count("processor\t:")
        mem = Path("/proc/meminfo").read_text().splitlines()[0]
        info["memory_kb"] = int(mem.split()[1])
    except (OSError, StopIteration, ValueError):
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    commit = git_commit(Path(__file__).resolve().parents[1])  # the code that trains
    cfg = load_config(args.config)
    log.info("config: %s", json.dumps(cfg.to_dict()))

    cols = ["notice_id", "title", "description", "language", "cpv_division", "cpv_group"]
    train = load_split(args.data / "train.parquet", cols)
    validation = load_split(args.data / "validation.parquet", cols)
    log.info("train %d rows, validation %d rows", len(train), len(validation))

    t0 = time.perf_counter()
    result = fit(train, cfg, validation=validation, log=log.info)
    total = time.perf_counter() - t0

    stats = json.loads((args.data / "stats.json").read_text())
    result.model.metadata.update(
        {
            "name": "eurotender-cpv-baseline",
            "package_version": eurotender.__version__,
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit,
            "dataset": {
                "files_sha256": stats.get("files"),
                "splits": stats.get("splits"),
                "train_rows_used": result.n_train,
            },
            "timings_seconds": {**result.timings, "total": total},
            "nnz_per_doc": result.nnz_per_doc,
            "versions": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "hardware": hardware(),
        }
    )
    result.model.save(args.out)
    log.info("saved model to %s in %.0fs", args.out, total)


if __name__ == "__main__":
    main()
