"""Orchestrates the four-system (Base | Short-SFT | Process-SFT | Full-MedLook)
evaluation report: answer quality on held-out VQA sets, and strategy + calibration on
the held-out gold strategy set.

Works identically on mocked predictions (local, no GPU -- see scripts/eval.py --mock and
tests/test_metrics.py) and on real model generations (Colab, after running inference
and collecting raw text outputs). The only difference is where `PredictionRecord.raw_generation`
comes from.

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions. Never invents or assumes lift -- `check_success_gate` only reports what the
numbers actually show.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from medlook.eval.answer import GoldAnswer, exact_match, score_answers
from medlook.eval.calibration import calibration_report
from medlook.eval.strategy import score_strategy
from medlook.schema import Action, MedLookResponse, SchemaError, parse

SYSTEM_ORDER = ("base", "short_sft", "process_sft", "full_medlook")
SYSTEM_LABELS = {
    "base": "Base (zero-shot)",
    "short_sft": "Short-SFT",
    "process_sft": "Process-SFT",
    "full_medlook": "Full-MedLook",
}


@dataclass
class PredictionRecord:
    """One scored example: a raw model generation plus the gold reference(s) needed to
    score it. `gold_action` is only ever populated for held-out gold-strategy-set
    records (see medlook/data/gold_strategy_set.py) -- never for the open VQA answer-
    quality sets, which have no hand-labeled strategy."""

    id: str
    question: str
    gold_answer: GoldAnswer
    raw_generation: str
    gold_action: Optional[Union[Action, str]] = None


def _fallback_response(raw_generation: str) -> MedLookResponse:
    """Used when a raw generation has no [FINAL] block at all -- e.g. a genuine
    zero-shot Base completion that never learned the MedLook schema. The system is
    still graded on the answer it actually gave (the whole raw generation), with a
    neutral default confidence -- Base has no learned calibration signal, so its
    calibration metrics are expected (and honest) to look uninformative rather than
    being unfairly zeroed out on ANSWER QUALITY just because it never emitted a
    schema tag it was never trained to produce."""
    text = raw_generation.strip()
    return MedLookResponse(final_answer=text if text else "(empty generation)", final_confidence=0.5)


def _safe_parse(raw_generation: str) -> Tuple[MedLookResponse, bool]:
    """Non-strict parse. Returns (response, used_fallback) -- `used_fallback` is True
    when there was no [FINAL] block at all and `_fallback_response` was used instead."""
    try:
        return parse(raw_generation, strict=False), False
    except SchemaError:
        return _fallback_response(raw_generation), True


def score_answer_quality(records: Sequence[PredictionRecord]) -> dict:
    preds, golds = [], []
    n_schema_fallback = 0
    for r in records:
        response, used_fallback = _safe_parse(r.raw_generation)
        n_schema_fallback += int(used_fallback)
        preds.append(response.final_answer)
        golds.append(r.gold_answer)
    report = score_answers(preds, golds).to_dict()
    report["n_schema_fallback"] = n_schema_fallback
    return report


def score_strategy_and_calibration(records: Sequence[PredictionRecord]) -> dict:
    """Requires every record to carry a `gold_action` (i.e. these must be gold
    strategy set records, never the open VQA answer-quality sets). A record whose
    generation has no [STRATEGY] block (short_sft/process_sft profiles, or Base, which
    never learned one) is scored as action 'UNPARSEABLE' -- always wrong against
    whatever the real gold_action is, which correctly penalizes systems that cannot
    express a strategy decision at all."""
    missing = [r.id for r in records if r.gold_action is None]
    if missing:
        raise ValueError(f"score_strategy_and_calibration requires gold_action on every record; missing for {missing}")

    pred_actions: List[str] = []
    confidences: List[float] = []
    corrects: List[bool] = []
    answer_preds: List[str] = []
    answer_golds: List[GoldAnswer] = []
    n_schema_fallback = 0

    for r in records:
        response, used_fallback = _safe_parse(r.raw_generation)
        n_schema_fallback += int(used_fallback)
        pred_actions.append(response.strategy.action if response.has_strategy else "UNPARSEABLE")
        confidences.append(response.final_confidence)
        corrects.append(exact_match(response.final_answer, r.gold_answer) >= 1.0)
        answer_preds.append(response.final_answer)
        answer_golds.append(r.gold_answer)

    gold_actions = [r.gold_action for r in records]

    return {
        "strategy": score_strategy(pred_actions, gold_actions).to_dict(),
        "calibration": calibration_report(confidences, corrects).to_dict(),
        "answer_quality": score_answers(answer_preds, answer_golds).to_dict(),
        "n_schema_fallback": n_schema_fallback,
    }


def _load_jsonl_records(path: str) -> List[PredictionRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                PredictionRecord(
                    id=row["id"],
                    question=row.get("question", ""),
                    gold_answer=row["gold_answer"],
                    raw_generation=row["raw_generation"],
                    gold_action=row.get("gold_action"),
                )
            )
    return records


def load_predictions_from_dir(
    root_dir: str,
) -> Tuple[Dict[str, Dict[str, List[PredictionRecord]]], Dict[str, List[PredictionRecord]]]:
    """Loads real Colab-generated predictions from the on-disk layout:

        root_dir/answer_quality/<system>/<eval_set>.jsonl
        root_dir/gold_strategy/<system>.jsonl

    Each JSONL line is `{"id", "question", "gold_answer", "raw_generation", "gold_action"?}`.
    `<system>` must be one of base/short_sft/process_sft/full_medlook to line up with
    `build_report`, though extra system names are tolerated and simply reported as-is.
    """
    answer_quality: Dict[str, Dict[str, List[PredictionRecord]]] = {}
    aq_dir = os.path.join(root_dir, "answer_quality")
    if os.path.isdir(aq_dir):
        for system in sorted(os.listdir(aq_dir)):
            system_dir = os.path.join(aq_dir, system)
            if not os.path.isdir(system_dir):
                continue
            answer_quality[system] = {}
            for fname in sorted(os.listdir(system_dir)):
                if fname.endswith(".jsonl"):
                    set_name = fname[: -len(".jsonl")]
                    answer_quality[system][set_name] = _load_jsonl_records(os.path.join(system_dir, fname))

    gold_strategy: Dict[str, List[PredictionRecord]] = {}
    gs_dir = os.path.join(root_dir, "gold_strategy")
    if os.path.isdir(gs_dir):
        for fname in sorted(os.listdir(gs_dir)):
            if fname.endswith(".jsonl"):
                system = fname[: -len(".jsonl")]
                gold_strategy[system] = _load_jsonl_records(os.path.join(gs_dir, fname))

    return answer_quality, gold_strategy


def build_report(
    answer_quality_records: Dict[str, Dict[str, List[PredictionRecord]]],
    gold_strategy_records: Dict[str, List[PredictionRecord]],
) -> dict:
    """`answer_quality_records` is system -> eval_set_name -> records (e.g.
    {"full_medlook": {"vqa_rad": [...], "pathvqa": [...]}}). `gold_strategy_records` is
    system -> records, all scored against the SAME held-out gold strategy set."""
    systems = [s for s in SYSTEM_ORDER if s in answer_quality_records or s in gold_strategy_records]
    systems += [s for s in answer_quality_records if s not in systems]
    systems += [s for s in gold_strategy_records if s not in systems]

    report: dict = {"systems": {}}
    for system in systems:
        entry: dict = {}
        if system in answer_quality_records:
            entry["answer_quality"] = {
                set_name: score_answer_quality(recs) for set_name, recs in answer_quality_records[system].items()
            }
        if system in gold_strategy_records:
            entry["gold_strategy"] = score_strategy_and_calibration(gold_strategy_records[system])
        report["systems"][system] = entry

    report["success_gate"] = check_success_gate(report)
    return report


def check_success_gate(
    report: dict,
    candidate: str = "full_medlook",
    baselines: Sequence[str] = ("base", "short_sft"),
) -> dict:
    """Implements the hard success gate from PROJECT_BLUEPRINT.md / Master Instructions
    4.4: `candidate` must improve mean token F1 on at least one primary answer-quality
    set AND improve calibration (lower AURC or lower overconfident-error) -- both
    relative to EVERY baseline in `baselines` that is present in the report (i.e. it
    must beat Base AND Short-SFT, not just one of them). Never invents a pass: any
    baseline/metric missing from the report makes that comparison unavailable rather
    than assumed-passing.
    """
    result: dict = {"passed": False, "reasons": [], "details": {}}
    systems = report.get("systems", {})
    if candidate not in systems:
        result["reasons"].append(f"candidate system '{candidate}' missing from report")
        return result

    present_baselines = [b for b in baselines if b in systems]
    if not present_baselines:
        result["reasons"].append("no baseline systems present in report to compare against")
        return result

    cand_aq = systems[candidate].get("answer_quality", {})
    answer_gate_per_baseline = {}
    improved_sets_by_baseline: Dict[str, List[str]] = {}
    for baseline in present_baselines:
        base_aq = systems[baseline].get("answer_quality", {})
        improved_sets = [
            set_name
            for set_name, cand_metrics in cand_aq.items()
            if set_name in base_aq and cand_metrics["token_f1"] > base_aq[set_name]["token_f1"]
        ]
        improved_sets_by_baseline[baseline] = improved_sets
        answer_gate_per_baseline[baseline] = len(improved_sets) > 0
    result["details"]["answer_f1_improved_sets_by_baseline"] = improved_sets_by_baseline
    answer_gate_passed = bool(answer_gate_per_baseline) and all(answer_gate_per_baseline.values())

    cand_calib = systems[candidate].get("gold_strategy", {}).get("calibration")
    calibration_gate_per_baseline = {}
    calibration_detail_by_baseline: Dict[str, dict] = {}
    for baseline in present_baselines:
        base_calib = systems[baseline].get("gold_strategy", {}).get("calibration")
        if not cand_calib or not base_calib:
            continue
        aurc_improved = cand_calib["aurc"] < base_calib["aurc"]
        overconf_improved = cand_calib["overconfident_error_rate"] < base_calib["overconfident_error_rate"]
        calibration_detail_by_baseline[baseline] = {
            "aurc_improved": aurc_improved,
            "overconfident_error_improved": overconf_improved,
        }
        calibration_gate_per_baseline[baseline] = aurc_improved or overconf_improved
    result["details"]["calibration_detail_by_baseline"] = calibration_detail_by_baseline
    calibration_gate_passed = bool(calibration_gate_per_baseline) and all(calibration_gate_per_baseline.values())

    if not answer_gate_passed:
        result["reasons"].append(
            "answer token-F1 did not improve on >=1 primary set over EVERY present baseline"
        )
    if not calibration_gate_passed:
        result["reasons"].append(
            "AURC / overconfident-error did not improve over EVERY present baseline"
        )

    result["passed"] = answer_gate_passed and calibration_gate_passed
    return result


def render_markdown_table(report: dict) -> str:
    lines = [
        "| System | Answer EM | Answer F1 | ACTION Acc | ACTION Macro-F1 | FLAG Prec | FLAG Rec | ECE | Overconf-Err | AURC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for system, entry in report.get("systems", {}).items():
        label = SYSTEM_LABELS.get(system, system)
        aq_sets = entry.get("answer_quality", {})
        if aq_sets:
            n_total = sum(m["n"] for m in aq_sets.values()) or 1
            em = sum(m["exact_match"] * m["n"] for m in aq_sets.values()) / n_total
            f1 = sum(m["token_f1"] * m["n"] for m in aq_sets.values()) / n_total
            em_str, f1_str = f"{em:.3f}", f"{f1:.3f}"
        else:
            em_str = f1_str = "n/a"

        gs = entry.get("gold_strategy")
        if gs:
            strat = gs["strategy"]
            calib = gs["calibration"]
            acc_str = f"{strat['accuracy']:.3f}"
            macro_f1_str = f"{strat['macro_f1']:.3f}"
            flag_p_str = f"{strat['flag_precision']:.3f}"
            flag_r_str = f"{strat['flag_recall']:.3f}"
            ece_str = f"{calib['ece']:.3f}"
            overconf_str = f"{calib['overconfident_error_rate']:.3f}"
            aurc_str = f"{calib['aurc']:.3f}"
        else:
            acc_str = macro_f1_str = flag_p_str = flag_r_str = ece_str = overconf_str = aurc_str = "n/a"

        lines.append(
            f"| {label} | {em_str} | {f1_str} | {acc_str} | {macro_f1_str} | "
            f"{flag_p_str} | {flag_r_str} | {ece_str} | {overconf_str} | {aurc_str} |"
        )

    gate = report.get("success_gate", {})
    lines.append("")
    lines.append(f"**Primary success gate: {'PASSED' if gate.get('passed') else 'NOT PASSED'}**")
    for reason in gate.get("reasons", []):
        lines.append(f"- {reason}")
    return "\n".join(lines)
