"""Serialization for adaptive-cap dispatch decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sloserve.router.adaptive_clipping import (
    AdaptiveClipLevel,
    AdaptiveClipTrigger,
)
from sloserve.router.admission import AdaptiveAdmissionDecision


@dataclass(frozen=True, slots=True)
class AdaptiveCapDecisionRecord:
    """Persistable adaptive-cap fact with repetition identity."""

    request_id: str
    repetition_index: int
    decision_time_s: float
    q_inst: int
    q_bar: float
    old_level: AdaptiveClipLevel
    new_level: AdaptiveClipLevel
    selected_cap: int | None
    trigger_reason: AdaptiveClipTrigger

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("adaptive-cap request_id must not be empty")
        if self.repetition_index < 0:
            raise ValueError("adaptive-cap repetition_index must be non-negative")

    @classmethod
    def from_admission(
        cls,
        admission_decision: AdaptiveAdmissionDecision,
        *,
        repetition_index: int,
    ) -> AdaptiveCapDecisionRecord:
        """Attach repetition identity to one in-memory admission decision."""
        decision = admission_decision.decision
        return cls(
            request_id=admission_decision.request_id,
            repetition_index=repetition_index,
            decision_time_s=decision.decision_time_s,
            q_inst=decision.q_inst,
            q_bar=decision.q_bar,
            old_level=decision.old_level,
            new_level=decision.new_level,
            selected_cap=decision.selected_cap,
            trigger_reason=decision.trigger_reason,
        )


def write_adaptive_cap_decisions_jsonl(
    path: str | Path,
    decisions: tuple[AdaptiveCapDecisionRecord, ...],
) -> Path:
    """Write canonical adaptive-cap decision facts in dispatch order."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for decision in decisions:
            output.write(
                json.dumps(
                    {
                        "request_id": decision.request_id,
                        "repetition_index": decision.repetition_index,
                        "decision_time_s": decision.decision_time_s,
                        "q_inst": decision.q_inst,
                        "q_bar": decision.q_bar,
                        "old_level": decision.old_level.name,
                        "new_level": decision.new_level.name,
                        "selected_cap": decision.selected_cap,
                        "trigger_reason": decision.trigger_reason.value,
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            output.write("\n")
    return output_path
