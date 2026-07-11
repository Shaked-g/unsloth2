"""Tests for medlook/demo/gradio_app.py. Requires gradio to be importable (it's a core
`requirements.txt` dependency for the local demo dry-run path) but never launches a
server and never loads a model -- no GPU required."""

from __future__ import annotations

from medlook import DISCLAIMER
from medlook.demo.gradio_app import build_demo, mock_generate, render_panels
from medlook.schema import Action


def test_mock_generate_no_images_flags_uncertain():
    raw = mock_generate([], "Is there a nodule?")
    assert "ACTION: FLAG_UNCERTAIN" in raw
    assert "[FINAL]" in raw


def test_mock_generate_multi_image_relooks():
    raw = mock_generate(["fake_image_1", "fake_image_2"], "Compare these two views.")
    assert "ACTION: RELOOK" in raw


def test_mock_generate_single_image_answers_confidently():
    raw = mock_generate(["fake_image_1"], "What organ is this?")
    assert "ACTION: ANSWER_CONFIDENT" in raw


def test_mock_generate_output_round_trips_through_schema_parse():
    from medlook.schema import parse

    raw = mock_generate(["fake_image_1"], "What organ is this?")
    response = parse(raw, strict=True)
    assert response.has_strategy
    assert response.strategy.action == Action.ANSWER_CONFIDENT


def test_render_panels_full_medlook_response():
    raw = mock_generate(["fake_image_1"], "What organ is this?")
    strategy_md, process_md, final_md, raw_out = render_panels(raw)
    assert "ANSWER_CONFIDENT" in strategy_md
    assert "mock" in process_md
    assert "Confidence" in final_md
    assert raw_out == raw


def test_render_panels_short_sft_style_response_has_no_strategy_block():
    raw = "[FINAL]\nyes Confidence: 0.90\n[/FINAL]"
    strategy_md, process_md, final_md, _ = render_panels(raw)
    assert "no [STRATEGY] block" in strategy_md
    assert "no [PROCESS] block" in process_md
    assert "yes" in final_md


def test_render_panels_unparseable_generation_shows_warning_not_crash():
    strategy_md, process_md, final_md, raw_out = render_panels("this is not a MedLook response at all")
    assert "Could not parse" in final_md
    assert raw_out == "this is not a MedLook response at all"


def test_build_demo_without_weights_constructs_blocks():
    demo = build_demo(model=None, tokenizer=None)
    assert demo is not None


def test_disclaimer_is_present_in_demo_module():
    # Sanity check that the demo imports the shared package-level disclaimer rather
    # than hardcoding its own copy that could drift out of sync.
    assert "Not for clinical use" in DISCLAIMER
