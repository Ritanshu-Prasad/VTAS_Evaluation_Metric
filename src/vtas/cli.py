"""Command-line interface: ``vtas score IMAGE CAPTION`` and ``vtas download``."""

import argparse
import sys

from ._version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vtas",
        description="VTAS: Visual-Truth Alignment Score for image captions.",
    )
    parser.add_argument("--version", action="version", version=f"vtas {__version__}")
    sub = parser.add_subparsers(dest="command")

    score = sub.add_parser("score", help="Score one image-caption pair.")
    score.add_argument("image", help="Path to the image file.")
    score.add_argument("caption", nargs="+", help="Caption text to evaluate.")
    score.add_argument("--beta", type=float, default=1.0, help="F-beta parameter.")

    sub.add_parser("download", help="Download the required spaCy model.")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "download":
        from . import _spacy_model

        _spacy_model.download()
        return 0

    if args.command == "score":
        from . import _spacy_model

        _spacy_model.ensure()
        from .vtas import VTASEvaluator, _print_report

        caption = " ".join(args.caption)
        print(f"VTAS v{__version__}")
        evaluator = VTASEvaluator(beta=args.beta)
        _print_report(evaluator.score(image_path=args.image, caption=caption))
        return 0

    _build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
