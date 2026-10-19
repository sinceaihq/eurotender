"""Publish the model, dataset and demo Space to the Hugging Face Hub.

Uses the token from the local Hugging Face login (``hf auth login``); nothing
is read from the repository. Each artifact is uploaded in a single commit and
the model and dataset are tagged with the package version.

    python scripts/publish_hf.py --model models/eurotender-cpv-baseline \
        --cards hub --data data/processed --web web --web-model web/model
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

import eurotender

ORG = "sinceaihq"
MODEL_REPO = f"{ORG}/eurotender-cpv-baseline"
DATASET_REPO = f"{ORG}/eurotender-cpv"
SPACE_REPO = f"{ORG}/eurotender-cpv-demo"


def tag(api: HfApi, repo: str, repo_type: str, version: str) -> None:
    refs = api.list_repo_refs(repo, repo_type=repo_type)
    if any(t.name == version for t in refs.tags):
        print(f"{repo}: tag {version} already exists, leaving it")
        return
    api.create_tag(repo, tag=version, repo_type=repo_type, tag_message=f"EuroTender {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--web-model", type=Path, required=True)
    parser.add_argument("--only", choices=["model", "dataset", "space"], action="append")
    parser.add_argument("--private", action="store_true", help="create repos as private")
    args = parser.parse_args()
    version = f"v{eurotender.__version__}"
    api = HfApi()
    only = set(args.only or ["model", "dataset", "space"])

    if "dataset" in only:
        api.create_repo(DATASET_REPO, repo_type="dataset", private=args.private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            for split in ("train", "validation", "test"):
                shutil.copy(args.data / f"{split}.parquet", root / "data" / f"{split}.parquet")
            shutil.copy(args.data / "stats.json", root / "stats.json")
            shutil.copy(args.cards / "dataset" / "README.md", root / "README.md")
            api.upload_folder(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                folder_path=root,
                commit_message=f"EuroTender CPV dataset {version}",
            )
        tag(api, DATASET_REPO, "dataset", version)
        print(f"https://huggingface.co/datasets/{DATASET_REPO}")

    if "model" in only:
        api.create_repo(MODEL_REPO, repo_type="model", private=args.private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for src in [args.model / "config.json", *sorted(args.model.glob("*.safetensors"))]:
                shutil.copy(src, root / src.name)
            shutil.copy(args.cards / "model" / "README.md", root / "README.md")
            for extra in ("metrics.json", "RESULTS.md"):
                src = args.cards.parent / "results" / extra
                if src.exists():
                    (root / "evaluation").mkdir(exist_ok=True)
                    shutil.copy(src, root / "evaluation" / extra)
            api.upload_folder(
                repo_id=MODEL_REPO,
                repo_type="model",
                folder_path=root,
                commit_message=f"EuroTender CPV baseline {version}",
            )
        tag(api, MODEL_REPO, "model", version)
        print(f"https://huggingface.co/{MODEL_REPO}")

    if "space" in only:
        api.create_repo(
            SPACE_REPO, repo_type="space", space_sdk="static", private=args.private, exist_ok=True
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("index.html", "eurotender.js", "casefold-table.js"):
                shutil.copy(args.web / name, root / name)
            shutil.copy(args.web / "SPACE_README.md", root / "README.md")
            (root / "model").mkdir()
            for name in ("web-model.json", "web-model.bin"):
                shutil.copy(args.web_model / name, root / "model" / name)
            api.upload_folder(
                repo_id=SPACE_REPO,
                repo_type="space",
                folder_path=root,
                commit_message=f"EuroTender demo {version}",
            )
        print(f"https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
