"""Schema round-trip, parsing, and validation tests. No GPU, no model weights."""

from __future__ import annotations

import pytest

from medlook import schema


def test_render_full_medlook_roundtrip():
    resp = schema.make_response(
        final_answer="There is a spiculated nodule in the upper lobe.",
        final_confidence=0.82,
        action=schema.Action.RELOOK,
        conf=0.6,
        reason="Margins were unclear at first glance; zoomed into the region.",
        process="Zoomed into the upper lobe region and re-examined the margin.",
    )
    text = schema.render_and_validate(resp)

    assert "[STRATEGY]" in text
    assert "ACTION: RELOOK" in text
    assert "[PROCESS]" in text
    assert "[FINAL]" in text
    assert "Confidence: 0.82" in text

    parsed = schema.parse(text)
    assert parsed.strategy.action == schema.Action.RELOOK
    assert parsed.strategy.conf == pytest.approx(0.6)
    assert parsed.process == resp.process
    assert parsed.final_answer == resp.final_answer
    assert parsed.final_confidence == pytest.approx(0.82)

    # Full round-trip: re-rendering the parsed response reproduces the same text.
    assert schema.render(parsed) == text


def test_short_sft_profile_has_no_strategy_or_process():
    resp = schema.make_response(final_answer="No acute abnormality.", final_confidence=0.91)
    text = schema.render(resp)

    assert "[STRATEGY]" not in text
    assert "[PROCESS]" not in text
    assert text.startswith("[FINAL]")

    parsed = schema.parse(text)
    assert parsed.strategy is None
    assert parsed.process is None
    assert parsed.final_answer == "No acute abnormality."


def test_process_sft_profile_has_process_but_no_strategy():
    resp = schema.MedLookResponse(
        final_answer="Mild cardiomegaly.",
        final_confidence=0.7,
        process="Compared cardiac width to thoracic width across the visible field.",
    )
    text = schema.render(resp)

    assert "[STRATEGY]" not in text
    assert "[PROCESS]" in text
    assert "[FINAL]" in text

    parsed = schema.parse(text)
    assert parsed.strategy is None
    assert parsed.process == resp.process


@pytest.mark.parametrize("action", list(schema.Action))
def test_all_actions_render_and_parse(action):
    resp = schema.make_response(
        final_answer="answer text",
        final_confidence=0.5,
        action=action,
        conf=0.5,
        reason="reason text",
    )
    text = schema.render(resp)
    parsed = schema.parse(text)
    assert parsed.strategy.action == action


def test_action_case_insensitive_on_parse():
    text = (
        "[STRATEGY]\n"
        "ACTION: flag_uncertain\n"
        "CONF: 0.30\n"
        "REASON: image too blurry\n"
        "[/STRATEGY]\n"
        "[FINAL]\n"
        "Cannot determine. Confidence: 0.30\n"
        "[/FINAL]"
    )
    parsed = schema.parse(text)
    assert parsed.strategy.action == schema.Action.FLAG_UNCERTAIN


def test_parse_missing_final_block_raises():
    with pytest.raises(schema.SchemaError):
        schema.parse("[STRATEGY]\nACTION: RELOOK\nCONF: 0.5\nREASON: x\n[/STRATEGY]\nno final here")


def test_parse_unknown_action_strict_raises():
    text = (
        "[STRATEGY]\nACTION: MAYBE_LOOK\nCONF: 0.5\nREASON: x\n[/STRATEGY]\n"
        "[FINAL]\nanswer Confidence: 0.5\n[/FINAL]"
    )
    with pytest.raises(schema.SchemaError):
        schema.parse(text, strict=True)


def test_parse_unknown_action_non_strict_drops_strategy():
    text = (
        "[STRATEGY]\nACTION: MAYBE_LOOK\nCONF: 0.5\nREASON: x\n[/STRATEGY]\n"
        "[FINAL]\nanswer Confidence: 0.5\n[/FINAL]"
    )
    parsed = schema.parse(text, strict=False)
    assert parsed.strategy is None
    assert parsed.final_answer == "answer"


def test_parse_missing_confidence_suffix_non_strict_defaults():
    text = "[FINAL]\njust an answer with no confidence\n[/FINAL]"
    parsed = schema.parse(text, strict=False)
    assert parsed.final_answer == "just an answer with no confidence"
    assert parsed.final_confidence == pytest.approx(0.5)


def test_parse_missing_confidence_suffix_strict_raises():
    text = "[FINAL]\njust an answer with no confidence\n[/FINAL]"
    with pytest.raises(schema.SchemaError):
        schema.parse(text, strict=True)


def test_validate_rejects_out_of_range_confidence():
    resp = schema.MedLookResponse(final_answer="x", final_confidence=1.5)
    with pytest.raises(schema.SchemaError):
        schema.validate(resp)


def test_validate_rejects_empty_answer():
    resp = schema.MedLookResponse(final_answer="   ", final_confidence=0.5)
    with pytest.raises(schema.SchemaError):
        schema.validate(resp)


def test_validate_rejects_empty_reason():
    resp = schema.MedLookResponse(
        final_answer="x",
        final_confidence=0.5,
        strategy=schema.Strategy(action=schema.Action.ESCALATE, conf=0.9, reason="   "),
    )
    with pytest.raises(schema.SchemaError):
        schema.validate(resp)


def test_make_response_requires_all_or_none_of_strategy_fields():
    with pytest.raises(schema.SchemaError):
        schema.make_response(final_answer="x", final_confidence=0.5, action="RELOOK")
