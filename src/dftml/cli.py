from __future__ import annotations

import argparse
import json

import pandas as pd

from .model import run_training
from .schema import validate_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dftml")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("data")
    train = sub.add_parser("train")
    train.add_argument("data")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        errors = validate_frame(pd.read_csv(args.data))
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print("Dataset is valid")
    else:
        print(json.dumps(run_training(args.data, args.config, args.output), indent=2))
    return 0
