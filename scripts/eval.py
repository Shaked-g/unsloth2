#!/usr/bin/env python
"""CLI: builds the four-system (Base | Short-SFT | Process-SFT | Full-MedLook)
evaluation report -- answer quality (EM/F1), strategy selection (ACTION accuracy/F1,
FLAG precision/recall), and calibration (ECE, overconfident-error, AURC).

Usage (local, no GPU -- proves the eval math works end to end on synthetic data):
    python scripts/eval.py --mock

Usage (real Colab generations, after running inference and dumping raw model outputs
via the on-disk layout documented in medlook.eval.runner.load_predictions_from_dir):
    python scripts/eval.py --predictions-dir path/to/predictions --out report.json

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions. --mock output is SYNTHETIC (built from tiny test fixtures with a scripted
correctness gradient) and must never be quoted as a real result -- it exists only to
validate that the metrics/report-building code is correct before spending Colab
compute on real generations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medlook.data.gold_strategy_set import load_gold_strategy_set
from medlook.eval.runner import (
    PredictionRecord,
    build_report,
    load_predictions_from_dir,
    render_markdown_table,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
MOCK_SYSTEM_CORRECT_RATES = {
    "base": 0.4,
    "short_sft": 0.6,
    "process_sft": 0.65,
    "full_medlook": 0.8,
}


def _mock_generation(profile: str, action: str, conf: float, answer: str) -> str:
    """Hand-assembles a raw MedLook-format string for a given ablation profile,
    deliberately WITHOUT calling medlook.schema.render -- this keeps --mock exercising
    the real parser (schema.parse) end to end, same as it would see an actual (possibly
    slightly malformed) model generation."""
    final = f"[FINAL]\n{answer} Confidence: {conf:.2f}\n[/FINAL]"
    if profile == "short_sft":
        return final
    process = "[PROCESS]\nConsidering the visible structures before answering.\n[/PROCESS]\n"
    if profile == "process_sft":
        return process + final
    strategy = f"[STRATEGY]\nACTION: {action}\nCONF: {conf:.2f}\nREASON: mock reasoning for {action}\n[/STRATEGY]\n"
    return strategy + final


def _make_raw_generation(system: str, correct: bool, action: str, gold_answer: str) -> str:
    conf = 0.9 if correct else 0.3  # honest: lower confidence exactly when wrong
    # The "incorrect" branch must actually produce a wrong answer, not just a lower
    # confidence -- otherwise EM/F1 would trivially be 1.0 regardless of `correct`.
    stated_answer = gold_answer if correct else "an unrelated finding not shown here"
    if system == "base":
        # Base never learned the MedLook schema -- realistic zero-shot completions
        # have no [FINAL] tag at all; the eval fallback path grades it on this raw text.
        return f"The answer is probably {stated_answer}." if correct else "I am not sure what this shows."
    return _mock_generation(system, action, conf, stated_answer)


def build_mock_predictions():
    with open(os.path.join(FIXTURES_DIR, "open_vqa_sample.json"), "r", encoding="utf-8") as f:
        aq_examples = json.load(f)
    gold_cases = load_gold_strategy_set(
        os.path.join(FIXTURES_DIR, "gold_strategy_sample.json"), os.path.join(FIXTURES_DIR, "images")
    )

    answer_quality: dict = {}
    gold_strategy: dict = {}
    for system, rate in MOCK_SYSTEM_CORRECT_RATES.items():
        aq_records = []
        for i, ex in enumerate(aq_examples):
            correct = (i / max(len(aq_examples) - 1, 1)) < rate
            raw = _make_raw_generation(system, correct, "ANSWER_CONFIDENT", ex["answer"])
            aq_records.append(
                PredictionRecord(id=ex["id"], question=ex["question"], gold_answer=ex["answer"], raw_generation=raw)
            )
        answer_quality[system] = {"mock_open_vqa": aq_records}

        gs_records = []
        for i, case in enumerate(gold_cases):
            correct = (i / max(len(gold_cases) - 1, 1)) < rate
            predicted_action = case.gold_action.value if (system == "full_medlook" and correct) else "ANSWER_CONFIDENT"
            raw = _make_raw_generation(system, correct, predicted_action, case.gold_answer)
            gs_records.append(
                PredictionRecord(
                    id=case.id,
                    question=case.question,
                    gold_answer=case.gold_answer,
                    raw_generation=raw,
                    gold_action=case.gold_action,
                )
            )
        gold_strategy[system] = gs_records

    return answer_quality, gold_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mock", action="store_true", help="Use synthetic mock predictions from fixtures (no GPU, no real model)"
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Directory of real prediction JSONL files (see medlook.eval.runner.load_predictions_from_dir)",
    )
    parser.add_argument("--out", default=None, help="Also write the JSON report to this path")
    args = parser.parse_args()

    if not args.mock and not args.predictions_dir:
        parser.error("must pass either --mock or --predictions-dir")

    if args.mock:
        print("=" * 78, file=sys.stderr)
        print("MOCK MODE: predictions are SYNTHETIC (built from test fixtures with a", file=sys.stderr)
        print("scripted correctness gradient). This proves the eval math/plumbing works;", file=sys.stderr)
        print("it is NOT a real result and must never be quoted as one.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        answer_quality, gold_strategy = build_mock_predictions()
    else:
        answer_quality, gold_strategy = load_predictions_from_dir(args.predictions_dir)

    report = build_report(answer_quality, gold_strategy)
    if args.mock:
        report["mock"] = True

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report to {args.out}", file=sys.stderr)

    print(render_markdown_table(report))


if __name__ == "__main__":
    main()
