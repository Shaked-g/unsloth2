"""Curriculum mixing and stratified-split tests. No network access, no GPU."""

from __future__ import annotations

from medlook.data import curriculum
from medlook.data.adapters.base import Record
from medlook.schema import Action, Strategy


def _make_record(id_, source, action, n_images=1):
    return Record(
        id=id_,
        images=[f"fake_image_{id_}"] * n_images,  # placeholder, curriculum doesn't touch pixels
        question=f"question for {id_}",
        strategy=Strategy(action=action, conf=0.7, reason="test"),
        final_answer="answer",
        final_confidence=0.7,
        source=source,
    )


def test_adapter_family_for_record():
    assert curriculum.adapter_family_for_record(_make_record("a", "vqa_rad", Action.RELOOK)) == "open_vqa"
    assert (
        curriculum.adapter_family_for_record(_make_record("b", "meissa_pathvqa", Action.RELOOK))
        == "meissa"
    )
    assert (
        curriculum.adapter_family_for_record(
            _make_record("c", "uncertainty_blur_vqa_rad", Action.FLAG_UNCERTAIN)
        )
        == "uncertainty"
    )


def test_mix_records_redistributes_weight_when_adapter_unavailable():
    records_by_adapter = {
        "open_vqa": [_make_record(f"ov{i}", "vqa_rad", Action.ANSWER_CONFIDENT) for i in range(10)],
        "meissa": [],  # Meissa-off mode
        "uncertainty": [_make_record(f"u{i}", "uncertainty_blur_x", Action.FLAG_UNCERTAIN) for i in range(10)],
    }
    config = curriculum.CurriculumConfig(
        weights={"open_vqa": 0.475, "meissa": 0.325, "uncertainty": 0.2}, seed=1
    )
    mixed = curriculum.mix_records(records_by_adapter, config, target_size=20)
    sources = {curriculum.adapter_family_for_record(r) for r in mixed}
    # meissa contributed nothing; the other two adapters absorbed its weight instead
    # of the dataset silently shrinking.
    assert "meissa" not in sources
    assert len(mixed) == 20


def test_mix_records_empty_when_all_adapters_unavailable():
    records_by_adapter = {"open_vqa": [], "meissa": [], "uncertainty": []}
    config = curriculum.CurriculumConfig()
    mixed = curriculum.mix_records(records_by_adapter, config)
    assert mixed == []


def test_mix_records_oversamples_when_pool_smaller_than_quota():
    records_by_adapter = {
        "open_vqa": [_make_record("ov0", "vqa_rad", Action.ANSWER_CONFIDENT)],
        "meissa": [],
        "uncertainty": [],
    }
    config = curriculum.CurriculumConfig(weights={"open_vqa": 1.0}, seed=1)
    mixed = curriculum.mix_records(records_by_adapter, config, target_size=5)
    assert len(mixed) == 5
    assert all(r.id == "ov0" for r in mixed)


def test_stratified_split_preserves_all_strata_in_both_splits_when_possible():
    records = []
    for action in (Action.ANSWER_CONFIDENT, Action.RELOOK, Action.FLAG_UNCERTAIN, Action.ESCALATE):
        for i in range(10):
            records.append(_make_record(f"{action.value}_{i}", "vqa_rad", action))

    train, val = curriculum.stratified_split(records, val_fraction=0.2, seed=1)
    assert len(train) + len(val) == len(records)

    train_actions = curriculum.action_histogram(train)
    val_actions = curriculum.action_histogram(val)
    # every action present in the source pool should appear in both splits
    for action in (Action.ANSWER_CONFIDENT, Action.RELOOK, Action.FLAG_UNCERTAIN, Action.ESCALATE):
        assert train_actions.get(action.value, 0) > 0
        assert val_actions.get(action.value, 0) > 0


def test_action_histogram_counts_correctly():
    records = [
        _make_record("a", "vqa_rad", Action.ANSWER_CONFIDENT),
        _make_record("b", "vqa_rad", Action.ANSWER_CONFIDENT),
        _make_record("c", "vqa_rad", Action.RELOOK),
    ]
    hist = curriculum.action_histogram(records)
    assert hist == {"ANSWER_CONFIDENT": 2, "RELOOK": 1}


def test_image_count_histogram():
    records = [
        _make_record("a", "vqa_rad", Action.RELOOK, n_images=1),
        _make_record("b", "vqa_rad", Action.RELOOK, n_images=2),
        _make_record("c", "vqa_rad", Action.RELOOK, n_images=2),
    ]
    hist = curriculum.image_count_histogram(records)
    assert hist == {"1": 1, "2": 2}
