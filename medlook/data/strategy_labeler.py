"""Strategy labeling heuristics for the `open_vqa` adapter.

Each rule below is a small, independently testable function that inspects a
(question, answer) pair and optionally returns a `LabelResult`. Rules run in a fixed
priority order in `label_question_answer`; the first rule to fire wins. Keeping each
heuristic isolated and documented means that if strategy labels look wrong for some
class of questions, the offending rule can be identified and fixed without touching
the others.

IMPORTANT (see PROJECT_BLUEPRINT.md Section 6 / "gold strategy evaluation set"): these
heuristics are used ONLY to generate *training* labels. They must never be used to
grade the model's strategy predictions -- that grading happens exclusively against the
held-out, hand-reviewed gold strategy set (`gold_strategy_set.py`). Evaluating against
these same heuristics would be circular and would make the strategy-selection metric
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from medlook.schema import Action

_HEDGE_WORDS = (
    "uncertain",
    "unclear",
    "cannot",
    "can't",
    "not sure",
    "ambiguous",
    "inconclusive",
    "indeterminate",
    "difficult to tell",
    "hard to tell",
)

_MULTI_HOP_KEYWORDS = (
    "how many",
    "compare",
    "comparison",
    "relative",
    "relationship",
    "explain",
    "both",
    "each",
    "difference between",
    "spatial",
    "which is larger",
    "which is bigger",
    "and how",
    "and explain",
    "and note",
)

_SIMPLE_QUESTION_STARTS = (
    "is ",
    "are ",
    "does ",
    "do ",
    "was ",
    "were ",
    "has ",
    "have ",
)


@dataclass
class LabelResult:
    action: Action
    conf: float
    reason: str


def _word_count(text: str) -> int:
    return len(text.strip().split())


def rule_hedge_in_answer(question: str, answer: str) -> Optional[LabelResult]:
    """If the gold answer itself contains hedge language, the question is inherently
    ambiguous in the source dataset -- label FLAG_UNCERTAIN rather than manufacturing
    confidence the underlying data doesn't support."""
    lowered = answer.lower()
    if any(hedge in lowered for hedge in _HEDGE_WORDS):
        return LabelResult(
            action=Action.FLAG_UNCERTAIN,
            conf=0.35,
            reason="Gold answer itself expresses uncertainty or ambiguity.",
        )
    return None


def rule_multi_hop_question(question: str, answer: str) -> Optional[LabelResult]:
    """Comparative, counting, or multi-region questions benefit from an explicit
    re-examination step rather than a single glance."""
    lowered = question.lower()
    if any(kw in lowered for kw in _MULTI_HOP_KEYWORDS):
        return LabelResult(
            action=Action.RELOOK,
            conf=0.6,
            reason="Question requires comparing or counting across multiple regions.",
        )
    return None


def rule_simple_closed_question(question: str, answer: str) -> Optional[LabelResult]:
    """Short yes/no or single-entity questions with short gold answers are the
    clearest ANSWER_CONFIDENT case."""
    lowered_q = question.strip().lower()
    is_closed_form = lowered_q.startswith(_SIMPLE_QUESTION_STARTS)
    is_short_answer = _word_count(answer) <= 3
    if is_closed_form and is_short_answer:
        return LabelResult(
            action=Action.ANSWER_CONFIDENT,
            conf=0.9,
            reason="Closed yes/no-style question with a short, unambiguous gold answer.",
        )
    return None


def rule_short_answer_fallback(question: str, answer: str) -> Optional[LabelResult]:
    """Fallback for short answers that didn't match the closed-question phrasing rule
    above (e.g. 'What organ is shown?' -> 'lung')."""
    if _word_count(answer) <= 3:
        return LabelResult(
            action=Action.ANSWER_CONFIDENT,
            conf=0.8,
            reason="Short, single-entity gold answer.",
        )
    return None


def rule_long_open_ended_answer(question: str, answer: str) -> Optional[LabelResult]:
    """Long, free-text answers (multi-clause finding descriptions) typically reflect a
    more involved visual read than a single-word closed answer."""
    if _word_count(answer) >= 12:
        return LabelResult(
            action=Action.RELOOK,
            conf=0.55,
            reason="Gold answer is a long, multi-clause finding description.",
        )
    return None


RuleFn = Callable[[str, str], Optional[LabelResult]]

_RULES: Tuple[RuleFn, ...] = (
    rule_hedge_in_answer,
    rule_multi_hop_question,
    rule_simple_closed_question,
    rule_short_answer_fallback,
    rule_long_open_ended_answer,
)


def label_question_answer(question: str, answer: str) -> LabelResult:
    """Apply the rule chain in priority order; the first match wins.

    If no rule fires (should be rare given the fallback rules), defaults to a
    mid-confidence RELOOK -- when heuristics genuinely don't know, biasing toward
    "look again" is safer than a false ANSWER_CONFIDENT.
    """
    for rule in _RULES:
        result = rule(question, answer)
        if result is not None:
            return result
    return LabelResult(
        action=Action.RELOOK,
        conf=0.5,
        reason="No labeling heuristic matched confidently; defaulting to re-examination.",
    )
