"""Tests for filter.py, decontaminate.py, convert.py, gold_strategy_set.py, and
serialize.py. No network access, no GPU."""

from __future__ import annotations

import os

from PIL import Image

from medlook.data import convert, decontaminate, filter as data_filter, serialize
from medlook.data.adapters.base import Record
from medlook.data.gold_strategy_set import action_distribution, load_gold_strategy_set
from medlook.schema import Action, Strategy

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
IMAGES_DIR = os.path.join(FIXTURES_DIR, "images")
GOLD_FIXTURE = os.path.join(FIXTURES_DIR, "gold_strategy_sample.json")


def _tiny_image(color=(1, 2, 3)):
    return Image.new("RGB", (8, 8), color=color)


def _make_record(
    id_,
    action=Action.ANSWER_CONFIDENT,
    answer="yes",
    n_images=1,
    question="q?",
    image_color=(1, 2, 3),
):
    return Record(
        id=id_,
        images=[_tiny_image(image_color) for _ in range(n_images)],
        question=question,
        strategy=Strategy(action=action, conf=0.7, reason="test reason"),
        final_answer=answer,
        final_confidence=0.7,
        source="vqa_rad",
    )


# --- filter.py ---


def test_apply_quality_filters_rejects_empty_answer():
    records = [_make_record("a", answer="")]
    kept, stats = data_filter.apply_quality_filters(records)
    assert kept == []
    assert stats.rejected_empty_answer == 1


def test_apply_quality_filters_rejects_too_many_images():
    records = [_make_record("a", n_images=data_filter.MAX_IMAGES_PER_SAMPLE + 1)]
    kept, stats = data_filter.apply_quality_filters(records)
    assert kept == []
    assert stats.rejected_too_many_images == 1


def test_apply_quality_filters_keeps_valid_records_and_counts_actions():
    records = [
        _make_record("a", action=Action.ANSWER_CONFIDENT),
        _make_record("b", action=Action.RELOOK),
        _make_record("c", action=Action.RELOOK),
    ]
    kept, stats = data_filter.apply_quality_filters(records)
    assert len(kept) == 3
    assert stats.action_counts == {"ANSWER_CONFIDENT": 1, "RELOOK": 2}


def test_check_class_balance_warns_when_flag_escalate_underrepresented():
    action_counts = {"ANSWER_CONFIDENT": 90, "RELOOK": 8, "FLAG_UNCERTAIN": 1, "ESCALATE": 1}
    warnings = data_filter.check_class_balance(action_counts)
    assert len(warnings) == 1


def test_check_class_balance_silent_when_balanced():
    action_counts = {"ANSWER_CONFIDENT": 40, "RELOOK": 30, "FLAG_UNCERTAIN": 15, "ESCALATE": 15}
    warnings = data_filter.check_class_balance(action_counts)
    assert warnings == []


# --- decontaminate.py ---


def test_filter_contaminated_flags_high_text_overlap():
    eval_records = [_make_record("gold_a", question="Is a nodule present in the lung?", answer="yes")]
    eval_grams, eval_hashes = decontamination_signature(eval_records)

    train_records = [_make_record("train_a", question="Is a nodule present in the lung?", answer="yes")]
    kept, report = decontaminate.filter_contaminated(train_records, eval_grams, eval_hashes)
    assert kept == []
    assert report.flagged_text_overlap == 1


def test_filter_contaminated_does_not_flag_generic_boilerplate_overlap():
    # Note: flat solid-color placeholder images all share an identical perceptual
    # hash (phash discards overall brightness/color and looks at texture, so a
    # textureless square gives no signal either way) -- this test uses real fixture
    # images with distinct shapes so the image-hash channel cannot coincidentally
    # cause or mask the result we're actually checking (text-overlap behavior).
    gold_image = Image.open(os.path.join(IMAGES_DIR, "gold_circle.png"))
    train_image = Image.open(os.path.join(IMAGES_DIR, "clean_square.png"))

    eval_records = [
        _make_record(
            "gold_a",
            question="Does the scan show a rounded opacity?",
            answer="Uncertain due to image quality; the opacity cannot be confidently confirmed.",
        )
    ]
    eval_records[0].images = [gold_image]
    eval_grams, eval_hashes = decontamination_signature(eval_records)

    # Shares the boilerplate uncertainty phrase but is about a completely different
    # question, and has an unrelated image -- should NOT be flagged as contamination
    # (regression test for the single-shared-5-gram false positive fixed during the
    # smoke run).
    train_records = [
        _make_record(
            "train_a",
            question="Is the cardiac silhouette enlarged on this view?",
            answer="Uncertain due to image quality; cannot confidently confirm or exclude the finding.",
        )
    ]
    train_records[0].images = [train_image]

    kept, report = decontaminate.filter_contaminated(train_records, eval_grams, eval_hashes)
    assert len(kept) == 1
    assert report.flagged_text_overlap == 0
    assert report.flagged_image_overlap == 0


def test_filter_contaminated_flags_image_hash_overlap():
    # Reuse the exact same real (textured) fixture image on both sides -- unlike a
    # flat placeholder color, this gives a genuine, distinguishing perceptual hash.
    shared_image = Image.open(os.path.join(IMAGES_DIR, "clean_cross.png"))
    eval_hashes = decontaminate.image_hashes([shared_image])

    train_record = _make_record("train_a", question="unrelated question", answer="unrelated answer")
    train_record.images = [shared_image]
    kept, report = decontaminate.filter_contaminated([train_record], set(), eval_hashes)
    assert kept == []
    assert report.flagged_image_overlap == 1


def decontamination_signature(records):
    texts = [f"{r.question} {r.final_answer}" for r in records]
    images_list = [r.images for r in records]
    return decontaminate.build_eval_signature_from_texts_images(texts, images_list)


# --- convert.py ---


def test_record_to_unsloth_sample_short_sft_profile():
    record = _make_record("a", action=Action.RELOOK)
    sample = convert.record_to_unsloth_sample(record, profile="short_sft")
    assistant_text = sample["messages"][1]["content"][0]["text"]
    assert "[STRATEGY]" not in assistant_text
    assert "[FINAL]" in assistant_text
    assert sample["meta"]["action"] == "RELOOK"


def test_record_to_unsloth_sample_full_medlook_profile_includes_strategy():
    record = _make_record("a", action=Action.FLAG_UNCERTAIN)
    sample = convert.record_to_unsloth_sample(record, profile="full_medlook")
    assistant_text = sample["messages"][1]["content"][0]["text"]
    assert "[STRATEGY]" in assistant_text
    assert "ACTION: FLAG_UNCERTAIN" in assistant_text


def test_record_to_unsloth_sample_user_content_has_question_and_images():
    record = _make_record("a", n_images=2)
    sample = convert.record_to_unsloth_sample(record)
    user_content = sample["messages"][0]["content"]
    assert user_content[0]["type"] == "text"
    image_items = [c for c in user_content if c["type"] == "image"]
    assert len(image_items) == 2


def test_records_to_unsloth_dataset_is_a_plain_list():
    records = [_make_record("a"), _make_record("b")]
    dataset = convert.records_to_unsloth_dataset(records)
    assert isinstance(dataset, list)
    assert len(dataset) == 2


# --- gold_strategy_set.py ---


def test_load_gold_strategy_set_returns_all_fixture_cases():
    cases = load_gold_strategy_set(GOLD_FIXTURE, IMAGES_DIR)
    assert len(cases) == 8
    dist = action_distribution(cases)
    assert dist == {"RELOOK": 2, "ANSWER_CONFIDENT": 2, "FLAG_UNCERTAIN": 2, "ESCALATE": 2}


# --- serialize.py ---


def test_save_and_load_dataset_jsonl_roundtrip(tmp_path):
    records = [_make_record("a", action=Action.RELOOK, n_images=2)]
    samples = convert.records_to_unsloth_dataset(records)

    jsonl_path = serialize.save_dataset_jsonl(samples, str(tmp_path), "train")
    assert os.path.exists(jsonl_path)

    loaded = serialize.load_dataset_jsonl(jsonl_path)
    assert len(loaded) == 1
    user_content = loaded[0]["messages"][0]["content"]
    image_items = [c for c in user_content if c["type"] == "image"]
    assert len(image_items) == 2
    for item in image_items:
        assert hasattr(item["image"], "size")  # materialized back into a PIL Image
