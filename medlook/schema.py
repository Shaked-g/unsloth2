"""MedLook structured response schema — single source of truth.

This module owns formatting, parsing, and validation of the MedLook output format:

    [STRATEGY]
    ACTION: RELOOK | ANSWER_CONFIDENT | FLAG_UNCERTAIN | ESCALATE
    CONF: 0.00-1.00
    REASON: short explanation
    [/STRATEGY]
    [PROCESS]
    ... optional multi-step / multi-image reasoning ...
    [/PROCESS]
    [FINAL]
    Answer. Confidence: X.XX
    [/FINAL]

Three ablation profiles reuse this one schema:
  - short_sft:    [FINAL] only
  - process_sft:  [PROCESS] + [FINAL], no [STRATEGY]
  - full_medlook: [STRATEGY] + optional [PROCESS] + [FINAL]

Every part of the system that touches this output format (data conversion, training-time
target validation, evaluation parsing, and the Gradio demo) must import this module rather
than re-implementing parsing logic, so there is exactly one place that can be wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


class Action(str, Enum):
    RELOOK = "RELOOK"
    ANSWER_CONFIDENT = "ANSWER_CONFIDENT"
    FLAG_UNCERTAIN = "FLAG_UNCERTAIN"
    ESCALATE = "ESCALATE"


ACTIONS = tuple(a.value for a in Action)


class SchemaError(ValueError):
    """Raised when a MedLook response cannot be parsed or fails validation."""


@dataclass
class Strategy:
    action: Union[Action, str]
    conf: float
    reason: str


@dataclass
class MedLookResponse:
    final_answer: str
    final_confidence: float
    strategy: Optional[Strategy] = None
    process: Optional[str] = None

    @property
    def has_strategy(self) -> bool:
        return self.strategy is not None

    @property
    def has_process(self) -> bool:
        return self.process is not None and self.process.strip() != ""


_STRATEGY_BLOCK_RE = re.compile(
    r"\[STRATEGY\]\s*"
    r"ACTION:\s*(?P<action>[A-Za-z_]+)\s*"
    r"CONF:\s*(?P<conf>[0-9]*\.?[0-9]+)\s*"
    r"REASON:\s*(?P<reason>.*?)\s*"
    r"\[/STRATEGY\]",
    re.DOTALL,
)

_PROCESS_BLOCK_RE = re.compile(
    r"\[PROCESS\]\s*(?P<process>.*?)\s*\[/PROCESS\]",
    re.DOTALL,
)

_FINAL_BLOCK_RE = re.compile(
    r"\[FINAL\]\s*(?P<body>.*?)\s*\[/FINAL\]",
    re.DOTALL,
)

_FINAL_CONF_RE = re.compile(
    r"Confidence:\s*(?P<conf>[0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE,
)


def _fmt_conf(conf: float) -> str:
    return f"{conf:.2f}"


def _action_value(action: Union[Action, str]) -> str:
    return action.value if isinstance(action, Action) else str(action)


def render(response: MedLookResponse) -> str:
    """Render a MedLookResponse into the canonical text format.

    Blocks are emitted only if present on the response, so this single function
    covers all three ablation profiles (short_sft / process_sft / full_medlook).
    """
    parts: list[str] = []

    if response.strategy is not None:
        s = response.strategy
        parts.append(
            "[STRATEGY]\n"
            f"ACTION: {_action_value(s.action)}\n"
            f"CONF: {_fmt_conf(s.conf)}\n"
            f"REASON: {s.reason.strip()}\n"
            "[/STRATEGY]"
        )

    if response.has_process:
        parts.append(f"[PROCESS]\n{response.process.strip()}\n[/PROCESS]")

    parts.append(
        "[FINAL]\n"
        f"{response.final_answer.strip()} Confidence: {_fmt_conf(response.final_confidence)}\n"
        "[/FINAL]"
    )

    return "\n".join(parts)


def parse(text: str, *, strict: bool = True) -> MedLookResponse:
    """Parse canonical MedLook text back into a MedLookResponse.

    A [STRATEGY] block and a [PROCESS] block are both optional; [FINAL] is mandatory.
    With strict=True (default) malformed or missing required pieces raise SchemaError.
    With strict=False, best-effort parsing is used and malformed optional blocks are
    dropped rather than raising -- this is what the eval suite uses when scoring raw
    model generations that may not perfectly conform to the schema.
    """
    if text is None:
        raise SchemaError("Cannot parse None as a MedLook response")

    strategy: Optional[Strategy] = None
    strategy_match = _STRATEGY_BLOCK_RE.search(text)
    if strategy_match:
        action_raw = strategy_match.group("action").strip().upper()
        try:
            action = Action(action_raw)
        except ValueError as exc:
            if strict:
                raise SchemaError(
                    f"Unknown ACTION '{action_raw}'. Must be one of {ACTIONS}."
                ) from exc
            action = None
        if action is not None:
            try:
                conf = float(strategy_match.group("conf"))
            except ValueError as exc:
                if strict:
                    raise SchemaError("STRATEGY CONF is not a valid float") from exc
                conf = 0.0
            reason = strategy_match.group("reason").strip()
            strategy = Strategy(action=action, conf=conf, reason=reason)
    elif strict and "[STRATEGY]" in text:
        raise SchemaError("Found [STRATEGY] tag but block did not match expected format")

    process: Optional[str] = None
    process_match = _PROCESS_BLOCK_RE.search(text)
    if process_match:
        process = process_match.group("process").strip()
    elif strict and "[PROCESS]" in text:
        raise SchemaError("Found [PROCESS] tag but block did not match expected format")

    final_match = _FINAL_BLOCK_RE.search(text)
    if not final_match:
        raise SchemaError("Missing required [FINAL]...[/FINAL] block")

    body = final_match.group("body").strip()
    conf_match = _FINAL_CONF_RE.search(body)
    if conf_match:
        final_confidence = float(conf_match.group("conf"))
        final_answer = body[: conf_match.start()].strip()
    elif strict:
        raise SchemaError("FINAL block missing 'Confidence: X.XX' suffix")
    else:
        final_confidence = 0.5
        final_answer = body.strip()

    if not final_answer:
        raise SchemaError("FINAL answer text is empty")

    return MedLookResponse(
        final_answer=final_answer,
        final_confidence=final_confidence,
        strategy=strategy,
        process=process,
    )


def validate(response: MedLookResponse) -> None:
    """Raise SchemaError if response violates any structural invariant.

    Checked invariants:
      - final_answer is non-empty
      - final_confidence is within [0, 1]
      - if a strategy is present: action is a known Action, conf is within [0, 1],
        and reason is non-empty
    """
    if not response.final_answer or not response.final_answer.strip():
        raise SchemaError("final_answer must be non-empty")

    if not (0.0 <= response.final_confidence <= 1.0):
        raise SchemaError(f"final_confidence {response.final_confidence} out of [0, 1] range")

    if response.strategy is not None:
        s = response.strategy
        if not isinstance(s.action, Action) and s.action not in ACTIONS:
            raise SchemaError(f"strategy.action '{s.action}' is not one of {ACTIONS}")

        if not (0.0 <= s.conf <= 1.0):
            raise SchemaError(f"strategy.conf {s.conf} out of [0, 1] range")

        if not s.reason or not s.reason.strip():
            raise SchemaError("strategy.reason must be non-empty")


def render_and_validate(response: MedLookResponse) -> str:
    """Convenience: validate then render. Preferred entry point for the data converter."""
    validate(response)
    return render(response)


def make_response(
    final_answer: str,
    final_confidence: float,
    *,
    action: Optional[Union[Action, str]] = None,
    conf: Optional[float] = None,
    reason: Optional[str] = None,
    process: Optional[str] = None,
) -> MedLookResponse:
    """Convenience constructor used by adapters and the strategy labeler.

    Pass action/conf/reason together to include a [STRATEGY] block (full_medlook
    profile), or omit all three for a short_sft-style response. Validates before
    returning so malformed data is caught at construction time, not later.
    """
    strategy: Optional[Strategy] = None
    if action is not None or conf is not None or reason is not None:
        if action is None or conf is None or reason is None:
            raise SchemaError("action, conf, and reason must all be provided together")
        norm_action = action if isinstance(action, Action) else Action(str(action).upper())
        strategy = Strategy(action=norm_action, conf=conf, reason=reason)

    response = MedLookResponse(
        final_answer=final_answer,
        final_confidence=final_confidence,
        strategy=strategy,
        process=process,
    )
    validate(response)
    return response
