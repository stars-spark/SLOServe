"""Train and evaluate the lightweight output-length predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from sloserve.config import WorkloadConfig, load_config
from sloserve.experiments.sweep import load_sweep_config
from sloserve.workload.length_predictor import (
    LengthPredictor,
    build_training_pairs,
    evaluate_predictor_quality,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _load_workload(path: Path) -> WorkloadConfig:
    try:
        return load_sweep_config(path).base_config.workload
    except (ValidationError, ValueError) as sweep_error:
        try:
            return load_config(path).workload
        except (ValidationError, ValueError):
            raise sweep_error from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--n", required=True, type=_positive_int)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.train_seed == args.eval_seed:
        raise ValueError("train-seed and eval-seed must differ")
    workload = _load_workload(args.config)
    training_pairs = build_training_pairs(workload, train_seed=args.train_seed, count=args.n)
    held_out_pairs = build_training_pairs(workload, train_seed=args.eval_seed, count=args.n)
    predictor = LengthPredictor.train(training_pairs)
    quality = evaluate_predictor_quality(predictor, held_out_pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(args.out)
    print(
        f"learned: MAE={quality['learned_mae']:.6f}, "
        f"Pearson corr={quality['learned_pearson_corr']:.6f}"
    )
    print(
        f"naive advertised cap: MAE={quality['naive_mae']:.6f}, "
        f"Pearson corr={quality['naive_pearson_corr']:.6f}"
    )
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
