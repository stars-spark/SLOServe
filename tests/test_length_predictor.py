"""Tests for the lightweight output-length predictor."""

from pathlib import Path

import pytest

from sloserve.config import LengthModel, RealisticLengthConfig, WorkloadConfig, load_config
from sloserve.router.models import RequestClass
from sloserve.workload.length_predictor import (
    LengthPredictor,
    LengthTrainingPair,
    build_training_pairs,
    evaluate_predictor_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _realistic_workload() -> WorkloadConfig:
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload
    return WorkloadConfig.model_validate(
        {
            **base.model_dump(),
            "length_model": LengthModel.REALISTIC,
            "realistic_length": RealisticLengthConfig().model_dump(),
        }
    )


def test_learned_predictor_beats_advertised_cap_on_held_out_data() -> None:
    workload = _realistic_workload()
    training_pairs = build_training_pairs(workload, train_seed=999, count=2_000)
    held_out_pairs = build_training_pairs(workload, train_seed=20250824, count=2_000)

    quality = evaluate_predictor_quality(LengthPredictor.train(training_pairs), held_out_pairs)

    assert quality["learned_mae"] < quality["naive_mae"]
    assert quality["learned_pearson_corr"] > quality["naive_pearson_corr"]


def test_predictor_round_trips_and_uses_global_fallback(tmp_path: Path) -> None:
    pairs = (
        LengthTrainingPair(RequestClass.INTERACTIVE, 64, "short", 2048, 100),
        LengthTrainingPair(RequestClass.INTERACTIVE, 128, "short", 2048, 300),
        LengthTrainingPair(RequestClass.BATCH, 512, "long", 2048, 900),
    )
    predictor = LengthPredictor.train(pairs)
    path = tmp_path / "predictor.json"

    predictor.save(path)
    loaded = LengthPredictor.load(path)

    assert loaded.to_json() == predictor.to_json()
    assert loaded.predict(RequestClass.INTERACTIVE, 999, "short", 4096) == 200.0
    assert loaded.predict(RequestClass.BATCH, 999, "unseen", 4096) == 300.0


def test_training_pairs_require_realistic_workload() -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload

    with pytest.raises(ValueError, match="requires a realistic workload"):
        build_training_pairs(workload, train_seed=999, count=10)
