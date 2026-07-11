"""Answer-quality metrics: exact match (EM) and token F1, using standard SQuAD/VQA-style
text normalization (lowercase, strip punctuation/articles, collapse whitespace).

Works on plain strings only -- no model, no GPU, no network. Used both by the local
mock eval path and by the real Colab eval path (after `schema.parse` has already
extracted `final_answer` from a raw generation).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Union

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

GoldAnswer = Union[str, Sequence[str]]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCTUATION_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _gold_variants(gold: GoldAnswer) -> List[str]:
    return [gold] if isinstance(gold, str) else list(gold)


def exact_match(pred: str, gold: GoldAnswer) -> float:
    """1.0 if the normalized prediction matches ANY normalized gold variant, else 0.0."""
    norm_pred = normalize_answer(pred)
    return 1.0 if any(norm_pred == normalize_answer(g) for g in _gold_variants(gold)) else 0.0


def _token_f1_single(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    num_common = sum(min(c, gold_counts.get(t, 0)) for t, c in pred_counts.items())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, gold: GoldAnswer) -> float:
    """Best token F1 against any of the gold variants (multi-reference VQA style)."""
    pred_tokens = normalize_answer(pred).split()
    best = 0.0
    for g in _gold_variants(gold):
        gold_tokens = normalize_answer(g).split()
        best = max(best, _token_f1_single(pred_tokens, gold_tokens))
    return best


@dataclass
class AnswerQualityReport:
    n: int
    exact_match: float
    token_f1: float

    def to_dict(self) -> dict:
        return {"n": self.n, "exact_match": self.exact_match, "token_f1": self.token_f1}


def score_answers(preds: Iterable[str], golds: Iterable[GoldAnswer]) -> AnswerQualityReport:
    preds = list(preds)
    golds = list(golds)
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) must be the same length")
    if not preds:
        return AnswerQualityReport(n=0, exact_match=0.0, token_f1=0.0)

    em_scores = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1_scores = [token_f1(p, g) for p, g in zip(preds, golds)]
    return AnswerQualityReport(
        n=len(preds),
        exact_match=sum(em_scores) / len(em_scores),
        token_f1=sum(f1_scores) / len(f1_scores),
    )
