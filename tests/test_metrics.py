"""Tests for medlook/eval/{answer,strategy,calibration,runner}.py using fixed, hand
(and cross-)checked input vectors with known expected values. No GPU, no network."""

from __future__ import annotations

import pytest

from medlook.eval import answer, calibration, strategy
from medlook.eval.runner import PredictionRecord, build_report, check_success_gate, score_answer_quality, score_strategy_and_calibration
from medlook.schema import Action

# --- answer.py ---


def test_normalize_answer_strips_articles_punctuation_and_case():
    assert answer.normalize_answer("The Lung, Yes!") == "lung yes"


def test_exact_match_true_after_normalization():
    assert answer.exact_match("Yes", "yes") == 1.0
    assert answer.exact_match("A lung nodule.", "lung nodule") == 1.0


def test_exact_match_false_when_different():
    assert answer.exact_match("There are two nodules", "two nodules") == 0.0


def test_exact_match_true_if_any_gold_variant_matches():
    assert answer.exact_match("no", ["yes", "no"]) == 1.0


def test_token_f1_known_value():
    assert answer.token_f1("There are two nodules", "two nodules") == pytest.approx(0.6666666666666666)


def test_token_f1_perfect_match_is_one():
    assert answer.token_f1("lung nodule", "lung nodule") == 1.0


def test_token_f1_no_overlap_is_zero():
    assert answer.token_f1("cardiac", "pulmonary") == 0.0


def test_score_answers_averages_across_examples():
    report = answer.score_answers(["yes", "no"], ["yes", "yes"])
    assert report.n == 2
    assert report.exact_match == 0.5


def test_score_answers_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        answer.score_answers(["yes"], ["yes", "no"])


def test_score_answers_empty_is_zero_not_error():
    report = answer.score_answers([], [])
    assert report.n == 0
    assert report.exact_match == 0.0


# --- strategy.py ---


PREDS = ["RELOOK", "RELOOK", "ANSWER_CONFIDENT", "FLAG_UNCERTAIN", "ESCALATE", "ANSWER_CONFIDENT"]
GOLDS = ["RELOOK", "ANSWER_CONFIDENT", "ANSWER_CONFIDENT", "FLAG_UNCERTAIN", "ESCALATE", "ESCALATE"]


def test_score_strategy_known_accuracy_and_flag_metrics():
    report = strategy.score_strategy(PREDS, GOLDS)
    assert report.accuracy == pytest.approx(0.6666666666666666)
    assert report.macro_f1 == pytest.approx(0.7083333333333333)
    assert report.flag_precision == pytest.approx(1.0)
    assert report.flag_recall == pytest.approx(0.6666666666666666)


def test_score_strategy_per_class_metrics():
    report = strategy.score_strategy(PREDS, GOLDS)
    assert report.per_class["FLAG_UNCERTAIN"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert report.per_class["ANSWER_CONFIDENT"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_score_strategy_confusion_matrix_counts():
    report = strategy.score_strategy(PREDS, GOLDS)
    # gold=ESCALATE was predicted ANSWER_CONFIDENT once
    assert report.confusion["ESCALATE"]["ANSWER_CONFIDENT"] == 1
    assert report.confusion["ESCALATE"]["ESCALATE"] == 1


def test_score_strategy_treats_action_enum_and_string_interchangeably():
    report_str = strategy.score_strategy(["RELOOK"], ["RELOOK"])
    report_enum = strategy.score_strategy([Action.RELOOK], [Action.RELOOK])
    assert report_str.accuracy == report_enum.accuracy == 1.0


def test_score_strategy_unknown_predicted_action_is_always_wrong():
    report = strategy.score_strategy(["NOT_A_REAL_ACTION"], ["RELOOK"])
    assert report.accuracy == 0.0


def test_score_strategy_empty_is_zero_not_error():
    report = strategy.score_strategy([], [])
    assert report.n == 0
    assert report.accuracy == 0.0


def test_score_strategy_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        strategy.score_strategy(["RELOOK"], ["RELOOK", "ESCALATE"])


# --- calibration.py ---


CONFIDENCES = [1.0, 1.0, 0.6, 0.6, 0.6, 0.2]
CORRECTS = [True, True, True, False, False, False]


def test_expected_calibration_error_known_value():
    ece = calibration.expected_calibration_error(CONFIDENCES, CORRECTS, n_bins=10)
    assert ece == pytest.approx(0.16666666666666666)


def test_expected_calibration_error_zero_for_perfectly_calibrated():
    # All confidence 1.0 and all correct -> zero calibration error.
    assert calibration.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0


def test_overconfident_error_rate_known_value():
    rate = calibration.overconfident_error_rate(CONFIDENCES, CORRECTS, threshold=0.5)
    assert rate == pytest.approx(0.4)


def test_overconfident_error_rate_zero_when_no_high_confidence_predictions():
    assert calibration.overconfident_error_rate([0.1, 0.2], [False, False], threshold=0.9) == 0.0


def test_risk_coverage_curve_known_points():
    curve = calibration.risk_coverage_curve(CONFIDENCES, CORRECTS)
    assert curve == pytest.approx(
        [
            (1 / 6, 0.0),
            (2 / 6, 0.0),
            (3 / 6, 0.0),
            (4 / 6, 0.25),
            (5 / 6, 0.4),
            (1.0, 0.5),
        ]
    )


def test_area_under_risk_coverage_known_value():
    aurc = calibration.area_under_risk_coverage(CONFIDENCES, CORRECTS)
    assert aurc == pytest.approx(0.15)


def test_area_under_risk_coverage_zero_when_always_correct():
    assert calibration.area_under_risk_coverage([0.9, 0.8, 0.7], [True, True, True]) == 0.0


def test_calibration_report_bundles_all_metrics():
    report = calibration.calibration_report(CONFIDENCES, CORRECTS)
    assert report.n == 6
    assert report.ece == pytest.approx(0.16666666666666666)
    assert report.aurc == pytest.approx(0.15)
    # default threshold=0.8: only the two conf=1.0 predictions qualify, both correct.
    assert report.overconfident_error_rate == 0.0


# --- runner.py ---


def _pred(id_, gold_answer, raw_generation, gold_action=None):
    return PredictionRecord(id=id_, question="q?", gold_answer=gold_answer, raw_generation=raw_generation, gold_action=gold_action)


def test_score_answer_quality_parses_full_medlook_schema():
    records = [
        _pred("a", "yes", "[STRATEGY]\nACTION: ANSWER_CONFIDENT\nCONF: 0.90\nREASON: clear\n[/STRATEGY]\n[FINAL]\nyes Confidence: 0.90\n[/FINAL]"),
        _pred("b", "no", "[FINAL]\nno Confidence: 0.80\n[/FINAL]"),
    ]
    report = score_answer_quality(records)
    assert report["exact_match"] == 1.0
    assert report["n_schema_fallback"] == 0


def test_score_answer_quality_falls_back_gracefully_on_schema_free_text():
    records = [_pred("a", "yes", "The answer is probably yes.")]
    report = score_answer_quality(records)
    assert report["n_schema_fallback"] == 1
    # fallback grades on the raw text, not zeroing it out just for lacking a schema tag
    assert report["token_f1"] > 0.0


def test_score_strategy_and_calibration_requires_gold_action():
    records = [_pred("a", "yes", "[FINAL]\nyes Confidence: 0.9\n[/FINAL]", gold_action=None)]
    with pytest.raises(ValueError):
        score_strategy_and_calibration(records)


def test_score_strategy_and_calibration_end_to_end():
    records = [
        _pred(
            "a",
            "yes",
            "[STRATEGY]\nACTION: ANSWER_CONFIDENT\nCONF: 0.9\nREASON: clear\n[/STRATEGY]\n[FINAL]\nyes Confidence: 0.9\n[/FINAL]",
            gold_action=Action.ANSWER_CONFIDENT,
        ),
        _pred(
            "b",
            "no",
            "[STRATEGY]\nACTION: RELOOK\nCONF: 0.4\nREASON: unclear\n[/STRATEGY]\n[FINAL]\nyes Confidence: 0.4\n[/FINAL]",
            gold_action=Action.FLAG_UNCERTAIN,
        ),
    ]
    result = score_strategy_and_calibration(records)
    assert result["strategy"]["accuracy"] == 0.5  # first correct action, second wrong
    assert result["answer_quality"]["exact_match"] == 0.5  # first correct answer, second wrong
    assert result["calibration"]["n"] == 2


def test_build_report_success_gate_passes_when_candidate_strictly_better():
    answer_quality = {
        "base": {"set1": [_pred("a", "yes", "no answer schema at all")]},
        "short_sft": {"set1": [_pred("a", "yes", "[FINAL]\nno Confidence: 0.9\n[/FINAL]")]},
        "full_medlook": {"set1": [_pred("a", "yes", "[FINAL]\nyes Confidence: 0.9\n[/FINAL]")]},
    }
    gold_strategy = {
        "base": [_pred("g1", "yes", "no schema", gold_action=Action.ANSWER_CONFIDENT)],
        "short_sft": [_pred("g1", "yes", "[FINAL]\nno Confidence: 0.9\n[/FINAL]", gold_action=Action.ANSWER_CONFIDENT)],
        "full_medlook": [
            _pred(
                "g1",
                "yes",
                "[STRATEGY]\nACTION: ANSWER_CONFIDENT\nCONF: 0.9\nREASON: clear\n[/STRATEGY]\n[FINAL]\nyes Confidence: 0.9\n[/FINAL]",
                gold_action=Action.ANSWER_CONFIDENT,
            )
        ],
    }
    report = build_report(answer_quality, gold_strategy)
    assert report["success_gate"]["passed"] is True


def test_build_report_success_gate_fails_when_candidate_worse_on_answer_quality():
    answer_quality = {
        "base": {"set1": [_pred("a", "yes", "[FINAL]\nyes Confidence: 0.9\n[/FINAL]")]},
        "full_medlook": {"set1": [_pred("a", "yes", "[FINAL]\nno Confidence: 0.9\n[/FINAL]")]},
    }
    gold_strategy = {
        "base": [_pred("g1", "yes", "[FINAL]\nyes Confidence: 0.9\n[/FINAL]", gold_action=Action.ANSWER_CONFIDENT)],
        "full_medlook": [
            _pred(
                "g1",
                "yes",
                "[STRATEGY]\nACTION: ANSWER_CONFIDENT\nCONF: 0.9\nREASON: clear\n[/STRATEGY]\n[FINAL]\nyes Confidence: 0.9\n[/FINAL]",
                gold_action=Action.ANSWER_CONFIDENT,
            )
        ],
    }
    report = build_report(answer_quality, gold_strategy)
    assert report["success_gate"]["passed"] is False
    assert "answer" in report["success_gate"]["reasons"][0]


def test_check_success_gate_missing_candidate_is_reported_not_crashed():
    result = check_success_gate({"systems": {"base": {}}}, candidate="full_medlook")
    assert result["passed"] is False
    assert "missing" in result["reasons"][0]
