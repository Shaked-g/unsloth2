"""Quality gates and class-balance checking, applied after adapters produce `Record`s
and before curriculum mixing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from medlook.data.adapters.base import Record
from medlook.schema import Action, MedLookResponse
from medlook.schema import validate as validate_response

MAX_IMAGES_PER_SAMPLE = 3
MIN_ANSWER_CHARS = 1
MAX_ANSWER_CHARS = 2000
MAX_QUESTION_CHARS = 2000
MIN_FLAG_ESCALATE_FRACTION = 0.12


@dataclass
class FilterStats:
    total_in: int = 0
    total_out: int = 0
    rejected_empty_answer: int = 0
    rejected_too_many_images: int = 0
    rejected_length: int = 0
    rejected_schema_invalid: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "rejected_empty_answer": self.rejected_empty_answer,
            "rejected_too_many_images": self.rejected_too_many_images,
            "rejected_length": self.rejected_length,
            "rejected_schema_invalid": self.rejected_schema_invalid,
            "action_counts": dict(self.action_counts),
            "flag_escalate_fraction": flag_escalate_fraction(self.action_counts),
        }


def flag_escalate_fraction(action_counts: Dict[str, int]) -> float:
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    flagged = action_counts.get(Action.FLAG_UNCERTAIN.value, 0) + action_counts.get(
        Action.ESCALATE.value, 0
    )
    return flagged / total


def _passes_quality_gates(record: Record) -> Tuple[bool, str]:
    if not record.final_answer or not record.final_answer.strip():
        return False, "rejected_empty_answer"
    if len(record.images) == 0 or len(record.images) > MAX_IMAGES_PER_SAMPLE:
        return False, "rejected_too_many_images"
    if not (MIN_ANSWER_CHARS <= len(record.final_answer) <= MAX_ANSWER_CHARS):
        return False, "rejected_length"
    if len(record.question) > MAX_QUESTION_CHARS:
        return False, "rejected_length"
    try:
        response = MedLookResponse(
            final_answer=record.final_answer,
            final_confidence=record.final_confidence,
            strategy=record.strategy,
            process=record.process,
        )
        validate_response(response)
    except Exception:
        return False, "rejected_schema_invalid"
    return True, ""


def apply_quality_filters(records: List[Record]) -> Tuple[List[Record], FilterStats]:
    stats = FilterStats()
    kept: List[Record] = []
    for record in records:
        stats.total_in += 1
        ok, reason = _passes_quality_gates(record)
        if not ok:
            setattr(stats, reason, getattr(stats, reason) + 1)
            continue
        kept.append(record)
        action = (
            record.strategy.action.value
            if isinstance(record.strategy.action, Action)
            else str(record.strategy.action)
        )
        stats.action_counts[action] = stats.action_counts.get(action, 0) + 1

    stats.total_out = len(kept)
    return kept, stats


def check_class_balance(action_counts: Dict[str, int]) -> List[str]:
    """Returns human-readable warnings if class-balance requirements
    (PROJECT_BLUEPRINT.md Section 6) are not met. Does not raise -- the caller
    (`prepare_data.py`) decides whether to treat this as fatal. Accepts a plain
    action-name -> count dict so it can be applied both to raw per-adapter filter
    stats and to the post-curriculum-mix action histogram."""
    warnings = []
    fraction = flag_escalate_fraction(action_counts)
    if fraction < MIN_FLAG_ESCALATE_FRACTION:
        warnings.append(
            f"FLAG_UNCERTAIN + ESCALATE fraction is {fraction:.1%}, below the "
            f"{MIN_FLAG_ESCALATE_FRACTION:.0%} minimum. Consider adding more "
            f"uncertainty-adapter samples or reweighting the curriculum."
        )
    return warnings
