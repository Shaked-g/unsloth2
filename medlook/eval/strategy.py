"""Strategy-selection metrics: ACTION accuracy/F1 against the held-out gold strategy
set (see `medlook/data/gold_strategy_set.py` for why this must never be the same pool
used to generate training-time heuristic labels).

Also reports FLAG precision/recall separately (FLAG_UNCERTAIN and ESCALATE are the
rarest, highest-stakes classes -- see PROJECT_BLUEPRINT.md / Master Instructions 4.3)
and a full confusion matrix for diagnosing systematic strategy confusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union

from medlook.schema import ACTIONS, Action

ActionLike = Union[Action, str]
FLAG_CLASSES = (Action.FLAG_UNCERTAIN.value, Action.ESCALATE.value)


def _norm(action: ActionLike) -> str:
    return action.value if isinstance(action, Action) else str(action).upper()


def confusion_matrix(preds: List[ActionLike], golds: List[ActionLike]) -> Dict[str, Dict[str, int]]:
    """Returns matrix[gold_action][pred_action] = count. Includes an 'UNPARSEABLE'
    predicted-action bucket for predictions that aren't one of the four known actions
    (e.g. the model omitted [STRATEGY] entirely, as short_sft/process_sft/base do)."""
    matrix = {g: {p: 0 for p in ACTIONS + ("UNPARSEABLE",)} for g in ACTIONS}
    for pred, gold in zip(preds, golds):
        gold_norm = _norm(gold)
        pred_norm = _norm(pred)
        if pred_norm not in ACTIONS:
            pred_norm = "UNPARSEABLE"
        matrix[gold_norm][pred_norm] += 1
    return matrix


@dataclass
class StrategyReport:
    n: int
    accuracy: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]
    flag_precision: float
    flag_recall: float
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "flag_precision": self.flag_precision,
            "flag_recall": self.flag_recall,
            "confusion": self.confusion,
        }


def score_strategy(preds: List[ActionLike], golds: List[ActionLike]) -> StrategyReport:
    """`preds` may include values outside the four known actions (e.g. None, or a
    profile that never emits [STRATEGY]) -- those are always scored as wrong."""
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) must be the same length")
    n = len(golds)
    if n == 0:
        return StrategyReport(
            n=0, accuracy=0.0, macro_f1=0.0, per_class={}, flag_precision=0.0, flag_recall=0.0, confusion={}
        )

    norm_preds = [_norm(p) if _norm(p) in ACTIONS else "UNPARSEABLE" for p in preds]
    norm_golds = [_norm(g) for g in golds]

    correct = sum(1 for p, g in zip(norm_preds, norm_golds) if p == g)
    accuracy = correct / n

    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    for action in ACTIONS:
        tp = sum(1 for p, g in zip(norm_preds, norm_golds) if p == action and g == action)
        fp = sum(1 for p, g in zip(norm_preds, norm_golds) if p == action and g != action)
        fn = sum(1 for p, g in zip(norm_preds, norm_golds) if p != action and g == action)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[action] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)

    flag_tp = sum(1 for p, g in zip(norm_preds, norm_golds) if p in FLAG_CLASSES and g in FLAG_CLASSES)
    flag_fp = sum(1 for p, g in zip(norm_preds, norm_golds) if p in FLAG_CLASSES and g not in FLAG_CLASSES)
    flag_fn = sum(1 for p, g in zip(norm_preds, norm_golds) if p not in FLAG_CLASSES and g in FLAG_CLASSES)
    flag_precision = flag_tp / (flag_tp + flag_fp) if (flag_tp + flag_fp) > 0 else 0.0
    flag_recall = flag_tp / (flag_tp + flag_fn) if (flag_tp + flag_fn) > 0 else 0.0

    confusion = confusion_matrix(norm_preds, norm_golds)

    return StrategyReport(
        n=n,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        flag_precision=flag_precision,
        flag_recall=flag_recall,
        confusion=confusion,
    )
