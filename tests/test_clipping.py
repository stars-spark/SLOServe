"""Tests for length-aware max-token clipping at admission."""

from pathlib import Path

from sloserve.config import AdmissionControlConfig, ClipSource
from sloserve.router.clipping import OutputClipper
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.length_predictor import LengthPredictor, LengthTrainingPair


def _request(
    *,
    target: int = 1800,
    advertised: int = 2048,
    prompt_kind: str = "long",
) -> RequestEnvelope:
    return RequestEnvelope(
        request_id="request-0",
        sequence_id=0,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=0.0,
        input_tokens=64,
        max_output_tokens=target,
        deadline_time_s=10.0,
        advertised_cap_tokens=advertised,
        prompt_kind=prompt_kind,
    )


def test_disabled_clipper_preserves_request_object() -> None:
    request = _request()

    clipped = OutputClipper(AdmissionControlConfig()).apply(request)

    assert clipped is request
    assert clipped.effective_max_output_tokens == request.max_output_tokens
    assert clipped.clip_applied is False


def test_advertised_source_clips_only_above_threshold() -> None:
    clipper = OutputClipper(
        AdmissionControlConfig(
            clip_enabled=True,
            clip_max_tokens=1024,
            clip_source=ClipSource.ADVERTISED,
        )
    )

    clipped = clipper.apply(_request(target=1800, advertised=2048))
    visible_short = clipper.apply(_request(target=1800, advertised=1000))
    actually_short = clipper.apply(_request(target=800, advertised=2048))

    assert clipped.max_output_tokens == 1800
    assert clipped.effective_max_output_tokens == 1024
    assert clipped.clip_applied is True
    assert visible_short.backend_max_output_tokens is None
    assert actually_short.backend_max_output_tokens is None


def test_learned_source_uses_visible_features_not_true_target(tmp_path: Path) -> None:
    predictor = LengthPredictor.train(
        (
            LengthTrainingPair(RequestClass.INTERACTIVE, 64, "long", 2048, 1500),
            LengthTrainingPair(RequestClass.INTERACTIVE, 64, "short", 2048, 300),
        )
    )
    artifact = tmp_path / "predictor.json"
    predictor.save(artifact)
    clipper = OutputClipper(
        AdmissionControlConfig(
            clip_enabled=True,
            clip_max_tokens=1024,
            clip_source=ClipSource.LEARNED,
            clip_estimator_path=str(artifact),
        )
    )

    first = clipper.apply(_request(target=1800, prompt_kind="long"))
    second = clipper.apply(_request(target=1300, prompt_kind="long"))
    visible_short = clipper.apply(_request(target=1800, prompt_kind="short"))

    assert first.effective_max_output_tokens == 1024
    assert second.effective_max_output_tokens == 1024
    assert visible_short.effective_max_output_tokens == 1800
