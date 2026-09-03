"""Lightweight output-length prediction for realistic synthetic workloads."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sloserve.config import LengthModel, WorkloadConfig
from sloserve.router.models import RequestClass
from sloserve.workload.generator import generate_requests

_ARTIFACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LengthTrainingPair:
    """One scheduler-visible feature vector paired with its true output length."""

    request_class: RequestClass
    input_tokens: int
    prompt_kind: str | None
    advertised_cap_tokens: int | None
    target_tokens: int


def build_training_pairs(
    config: WorkloadConfig, *, train_seed: int, count: int
) -> tuple[LengthTrainingPair, ...]:
    """Generate labeled examples from a realistic workload under an independent seed."""
    if config.length_model is not LengthModel.REALISTIC:
        raise ValueError("length predictor training requires a realistic workload")
    if train_seed < 0:
        raise ValueError("train_seed must be non-negative")
    if count < 1:
        raise ValueError("count must be positive")
    training_config = WorkloadConfig.model_validate(
        {
            **config.model_dump(),
            "random_seed": train_seed,
            "total_requests": count,
            "warmup_requests": 0,
        }
    )
    return tuple(
        LengthTrainingPair(
            request_class=request.request_class,
            input_tokens=request.input_tokens,
            prompt_kind=request.prompt_kind,
            advertised_cap_tokens=request.advertised_cap_tokens,
            target_tokens=request.max_output_tokens,
        )
        for request in generate_requests(training_config)
    )


class LengthPredictor:
    """Median output length per request-class and prompt-kind bucket."""

    def __init__(
        self,
        *,
        bucket_medians: dict[tuple[str, str | None], float],
        global_median: float,
    ) -> None:
        if not bucket_medians:
            raise ValueError("length predictor requires at least one bucket")
        if global_median <= 0.0:
            raise ValueError("global median must be positive")
        self._bucket_medians = dict(bucket_medians)
        self.global_median = float(global_median)

    @classmethod
    def train(cls, pairs: Iterable[LengthTrainingPair]) -> LengthPredictor:
        """Fit bucket and fallback medians from labeled examples."""
        materialized = tuple(pairs)
        if not materialized:
            raise ValueError("length predictor training pairs must not be empty")
        buckets: dict[tuple[str, str | None], list[int]] = defaultdict(list)
        targets: list[int] = []
        for pair in materialized:
            if pair.target_tokens < 1:
                raise ValueError("training target_tokens must be positive")
            key = (pair.request_class.value, pair.prompt_kind)
            buckets[key].append(pair.target_tokens)
            targets.append(pair.target_tokens)
        return cls(
            bucket_medians={
                key: float(statistics.median(values)) for key, values in buckets.items()
            },
            global_median=float(statistics.median(targets)),
        )

    def predict(
        self,
        request_class: RequestClass,
        input_tokens: int,
        prompt_kind: str | None,
        advertised_cap_tokens: int | None,
    ) -> float:
        """Predict from the matching bucket, with a global fallback for unseen features."""
        del input_tokens, advertised_cap_tokens
        return self._bucket_medians.get(
            (request_class.value, prompt_kind),
            self.global_median,
        )

    def to_json(self) -> str:
        """Serialize the predictor to a deterministic, versioned JSON artifact."""
        buckets = [
            {
                "request_class": request_class,
                "prompt_kind": prompt_kind,
                "median_tokens": median,
            }
            for (request_class, prompt_kind), median in sorted(
                self._bucket_medians.items(), key=lambda item: (item[0][0], item[0][1] or "")
            )
        ]
        return (
            json.dumps(
                {
                    "artifact_version": _ARTIFACT_VERSION,
                    "global_median_tokens": self.global_median,
                    "bucket_medians": buckets,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def save(self, path: str | Path) -> None:
        """Write the predictor artifact to disk."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LengthPredictor:
        """Load and validate a serialized predictor artifact."""
        artifact_path = Path(path)
        try:
            raw: Any = json.loads(artifact_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"cannot read length predictor artifact: {artifact_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid length predictor JSON: {artifact_path}") from exc
        if not isinstance(raw, dict) or raw.get("artifact_version") != _ARTIFACT_VERSION:
            raise ValueError("unsupported length predictor artifact version")
        raw_buckets = raw.get("bucket_medians")
        global_median = raw.get("global_median_tokens")
        if not isinstance(raw_buckets, list) or not isinstance(global_median, (int, float)):
            raise ValueError("invalid length predictor artifact structure")
        bucket_medians: dict[tuple[str, str | None], float] = {}
        for bucket in raw_buckets:
            if not isinstance(bucket, dict):
                raise ValueError("invalid length predictor bucket")
            request_class = bucket.get("request_class")
            prompt_kind = bucket.get("prompt_kind")
            median = bucket.get("median_tokens")
            if (
                not isinstance(request_class, str)
                or (prompt_kind is not None and not isinstance(prompt_kind, str))
                or not isinstance(median, (int, float))
                or median <= 0.0
            ):
                raise ValueError("invalid length predictor bucket")
            RequestClass(request_class)
            bucket_medians[(request_class, prompt_kind)] = float(median)
        return cls(bucket_medians=bucket_medians, global_median=float(global_median))


def predict_advertised_cap(
    request_class: RequestClass,
    input_tokens: int,
    prompt_kind: str | None,
    advertised_cap_tokens: int | None,
) -> float:
    """Return the scheduler-visible cap as a naive constant prediction."""
    del request_class, input_tokens, prompt_kind
    if advertised_cap_tokens is None:
        raise ValueError("advertised-cap baseline requires advertised_cap_tokens")
    return float(advertised_cap_tokens)


def evaluate_predictor_quality(
    predictor: LengthPredictor, pairs: Sequence[LengthTrainingPair]
) -> dict[str, float]:
    """Report held-out MAE and Pearson correlation for learned and cap predictions."""
    if not pairs:
        raise ValueError("held-out evaluation pairs must not be empty")
    actual = [float(pair.target_tokens) for pair in pairs]
    learned = [
        predictor.predict(
            pair.request_class,
            pair.input_tokens,
            pair.prompt_kind,
            pair.advertised_cap_tokens or pair.target_tokens,
        )
        for pair in pairs
    ]
    naive = [
        predict_advertised_cap(
            pair.request_class,
            pair.input_tokens,
            pair.prompt_kind,
            pair.advertised_cap_tokens,
        )
        for pair in pairs
    ]
    return {
        "learned_mae": _mean_absolute_error(actual, learned),
        "learned_pearson_corr": _pearson_corr(actual, learned),
        "naive_mae": _mean_absolute_error(actual, naive),
        "naive_pearson_corr": _pearson_corr(actual, naive),
    }


def _mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return statistics.fmean(
        abs(observed - estimate) for observed, estimate in zip(actual, predicted, strict=True)
    )


def _pearson_corr(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_mean = statistics.fmean(actual)
    predicted_mean = statistics.fmean(predicted)
    actual_centered = [value - actual_mean for value in actual]
    predicted_centered = [value - predicted_mean for value in predicted]
    denominator = math.sqrt(
        sum(value * value for value in actual_centered)
        * sum(value * value for value in predicted_centered)
    )
    if denominator == 0.0:
        return 0.0
    return (
        sum(
            observed * estimate
            for observed, estimate in zip(actual_centered, predicted_centered, strict=True)
        )
        / denominator
    )
