"""Adapter and strategy-labeler tests. No network access, no GPU."""

from __future__ import annotations

import os

import pytest

from medlook.data.adapters.meissa import MeissaAdapter
from medlook.data.adapters.open_vqa import OpenVQAAdapter
from medlook.data.adapters.uncertainty import UncertaintyAdapter
from medlook.data.strategy_labeler import label_question_answer
from medlook.schema import Action

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
IMAGES_DIR = os.path.join(FIXTURES_DIR, "images")
OPEN_VQA_FIXTURE = os.path.join(FIXTURES_DIR, "open_vqa_sample.json")
MEISSA_FIXTURE = os.path.join(FIXTURES_DIR, "meissa_sample.json")


# --- strategy_labeler heuristics (each rule individually exercised) ---


def test_hedge_word_in_answer_flags_uncertain():
    result = label_question_answer("Is there a lesion?", "It is unclear from this view.")
    assert result.action == Action.FLAG_UNCERTAIN


def test_multi_hop_question_triggers_relook():
    result = label_question_answer(
        "How many nodules are visible and how do they compare in size?", "There are two."
    )
    assert result.action == Action.RELOOK


def test_simple_closed_question_short_answer_is_confident():
    result = label_question_answer("Is the heart enlarged?", "no")
    assert result.action == Action.ANSWER_CONFIDENT


def test_short_non_closed_answer_falls_back_to_confident():
    result = label_question_answer("What organ is shown?", "lung")
    assert result.action == Action.ANSWER_CONFIDENT


def test_long_open_ended_answer_triggers_relook():
    result = label_question_answer(
        "Describe the findings.",
        "A poorly differentiated infiltrating lesion is visible, disrupting the normal glandular architecture.",
    )
    assert result.action == Action.RELOOK


# --- open_vqa adapter ---


def test_open_vqa_adapter_from_fixture_is_available():
    adapter = OpenVQAAdapter.from_fixture(OPEN_VQA_FIXTURE, IMAGES_DIR)
    assert adapter.is_available()
    records = list(adapter.iter_records())
    assert len(records) == 6
    for r in records:
        assert len(r.images) == 1
        assert r.strategy.action in list(Action)
        assert r.final_answer


def test_open_vqa_adapter_labels_match_expected_actions():
    adapter = OpenVQAAdapter.from_fixture(OPEN_VQA_FIXTURE, IMAGES_DIR)
    records = {r.id: r for r in adapter.iter_records()}
    assert records["openvqa_0001"].strategy.action == Action.ANSWER_CONFIDENT  # yes/no, short
    assert records["openvqa_0003"].strategy.action == Action.RELOOK  # long open-ended
    assert records["openvqa_0004"].strategy.action == Action.RELOOK  # "how many" + compare


def test_open_vqa_adapter_missing_fixture_raises_not_silently_empty():
    # from_fixture is expected to raise on a missing REQUIRED (primary) source --
    # unlike meissa, open_vqa has no graceful-fallback contract.
    with pytest.raises(FileNotFoundError):
        OpenVQAAdapter.from_fixture(os.path.join(FIXTURES_DIR, "does_not_exist.json"), IMAGES_DIR)


# --- meissa adapter (optional, graceful fallback) ---


def test_meissa_adapter_from_fixture_is_available():
    adapter = MeissaAdapter.from_fixture(MEISSA_FIXTURE, IMAGES_DIR)
    assert adapter.is_available()
    records = list(adapter.iter_records())
    assert len(records) == 3


def test_meissa_adapter_tool_step_maps_to_relook_with_real_image():
    adapter = MeissaAdapter.from_fixture(MEISSA_FIXTURE, IMAGES_DIR)
    records = {r.id: r for r in adapter.iter_records()}
    tool_record = records["pathvqa_fixture_0001"]
    assert tool_record.strategy.action == Action.RELOOK
    assert tool_record.process is not None and len(tool_record.process) > 0
    assert len(tool_record.images) >= 1


def test_meissa_adapter_no_tool_step_maps_to_answer_confident():
    adapter = MeissaAdapter.from_fixture(MEISSA_FIXTURE, IMAGES_DIR)
    records = {r.id: r for r in adapter.iter_records()}
    direct_record = records["vqa_rad_fixture_0002"]
    assert direct_record.strategy.action == Action.ANSWER_CONFIDENT
    assert direct_record.process is None


def test_meissa_adapter_missing_fixture_is_gracefully_unavailable():
    adapter = MeissaAdapter.from_fixture(os.path.join(FIXTURES_DIR, "does_not_exist.json"), IMAGES_DIR)
    assert not adapter.is_available()
    assert list(adapter.iter_records()) == []


def test_meissa_adapter_malformed_json_is_gracefully_unavailable(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    adapter = MeissaAdapter.from_fixture(str(bad_path), IMAGES_DIR)
    assert not adapter.is_available()


# --- uncertainty adapter (anti-shortcut pairing) ---


def test_uncertainty_adapter_produces_four_variants_per_example():
    open_vqa_adapter = OpenVQAAdapter.from_fixture(OPEN_VQA_FIXTURE, IMAGES_DIR)
    base_records = list(open_vqa_adapter.iter_records())
    unc_adapter = UncertaintyAdapter.from_open_vqa_records(base_records)
    records = list(unc_adapter.iter_records())
    assert len(records) == 4 * len(base_records)


def test_uncertainty_adapter_pairing_shares_pair_id_and_diverges_only_in_variant():
    open_vqa_adapter = OpenVQAAdapter.from_fixture(OPEN_VQA_FIXTURE, IMAGES_DIR)
    base_records = list(open_vqa_adapter.iter_records())[:1]
    unc_adapter = UncertaintyAdapter.from_open_vqa_records(base_records)
    records = list(unc_adapter.iter_records())

    pair_ids = {r.extra["pair_id"] for r in records}
    assert pair_ids == {base_records[0].id}

    variants = {r.extra["variant"]: r for r in records}
    assert variants["clean"].strategy.action == Action.ANSWER_CONFIDENT
    assert variants["blur"].strategy.action == Action.FLAG_UNCERTAIN
    assert variants["crop"].strategy.action == Action.FLAG_UNCERTAIN
    assert variants["conflict"].strategy.action == Action.ESCALATE

    # Anti-shortcut invariant: all four variants share the same base question text
    # (the conflict variant appends an explicit note, but the core question matches).
    assert variants["clean"].question in variants["conflict"].question
    assert variants["clean"].question == variants["blur"].question == variants["crop"].question


def test_uncertainty_adapter_empty_input_is_unavailable():
    unc_adapter = UncertaintyAdapter.from_open_vqa_records([])
    assert not unc_adapter.is_available()
    assert list(unc_adapter.iter_records()) == []
