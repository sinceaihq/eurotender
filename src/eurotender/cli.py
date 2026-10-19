"""Command line interface: ``eurotender predict "text"``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .predict import load_model, predict_cpv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eurotender", description="CPV suggestions for EU tenders"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="rank CPV codes for one or more texts")
    p.add_argument("text", nargs="*", help="notice title/description; reads stdin lines if omitted")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--level", default="division", choices=["division", "group"])
    p.add_argument("--lang", default="en", help="label language, e.g. en, fi, de")
    p.add_argument("--model", default=None, help="local model directory or Hub repo id")
    p.add_argument("--revision", default=None, help="Hub revision (tag, branch or commit)")
    p.add_argument("--json", action="store_true", help="print JSON lines")
    args = parser.parse_args(argv)

    texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        parser.error("no input text")
    model = load_model(args.model, revision=args.revision)
    results = predict_cpv(texts, top_k=args.top_k, level=args.level, lang=args.lang, model=model)
    for text, preds in zip(texts, results, strict=True):
        if args.json:
            print(
                json.dumps(
                    {"text": text, "predictions": [x.to_dict() for x in preds]}, ensure_ascii=False
                )
            )
        else:
            print(text)
            for x in preds:
                print(f"  {x.code}  {x.score:6.3f}  {x.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
