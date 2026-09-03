#!/usr/bin/env python
"""Unified antibody robustness evaluation entrypoint."""

import argparse
import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate antibody robustness samples.")
    parser.add_argument("--dataset", required=True, help="Robustness benchmark to evaluate.")
    parser.add_argument("sample_result", help="Path to sample_humanization_result.csv")
    args = parser.parse_args()

    dataset_key = {
        "chicken": "chicken",
        "rabbit": "rabbit",
        "emicizumab": "Emicizumab",
        "Emicizumab": "Emicizumab",
        "BH1": "BH1",
        "HuAb348": "HuAb348",
        "Humab25": "Humab25",
    }.get(args.dataset)
    if dataset_key is None:
        raise ValueError(f"Unsupported antibody dataset: {args.dataset}")

    module_names = {
        "chicken": "chicken_eval",
        "rabbit": "rabbit_eval",
        "Emicizumab": "emicizumab_eval",
        "BH1": "2B04_eval",
        "HuAb348": "HuAb348_eval",
        "Humab25": "humab25_eval",
    }
    module = importlib.import_module(
        f"evaluation.Robustness.antibody_backends.{module_names[dataset_key]}"
    )
    module.main(args.sample_result)


if __name__ == "__main__":
    main()
