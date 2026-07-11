"""Decontamination: checks the training pool for overlap against held-out evaluation
data (VQA-RAD/SLAKE/PathVQA held-out subsets, and the gold strategy set), using both
text n-gram overlap and image perceptual-hash overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple

import imagehash
from PIL import Image

from medlook.data.adapters.base import Record

_WORD_RE = re.compile(r"[a-z0-9]+")
DEFAULT_NGRAM_SIZE = 5


def text_ngrams(text: str, n: int = DEFAULT_NGRAM_SIZE) -> Set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def image_hashes(images: Iterable[Image.Image]) -> Set[str]:
    return {str(imagehash.phash(img.convert("RGB"))) for img in images}


def build_eval_signature_from_texts_images(
    texts: Iterable[str], images_list: Iterable[Iterable[Image.Image]]
) -> Tuple[Set[str], Set[str]]:
    text_grams: Set[str] = set()
    hashes: Set[str] = set()
    for text, images in zip(texts, images_list):
        text_grams |= text_ngrams(text)
        hashes |= image_hashes(images)
    return text_grams, hashes


def build_eval_signature(eval_records: Iterable[Record]) -> Tuple[Set[str], Set[str]]:
    """Convenience wrapper for `Record`-shaped held-out pools."""
    texts = [f"{r.question} {r.final_answer}" for r in eval_records]
    images_list = [r.images for r in eval_records]
    return build_eval_signature_from_texts_images(texts, images_list)


@dataclass
class DecontaminationReport:
    total_checked: int = 0
    flagged_text_overlap: int = 0
    flagged_image_overlap: int = 0
    flagged_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        total = self.total_checked or 1
        return {
            "total_checked": self.total_checked,
            "flagged_text_overlap": self.flagged_text_overlap,
            "flagged_image_overlap": self.flagged_image_overlap,
            "decontamination_rate": (
                (self.flagged_text_overlap + self.flagged_image_overlap) / total
            ),
            "flagged_ids": self.flagged_ids,
        }


DEFAULT_TEXT_OVERLAP_THRESHOLD = 0.5


def filter_contaminated(
    records: Iterable[Record],
    eval_text_grams: Set[str],
    eval_image_hashes: Set[str],
    text_overlap_threshold: float = DEFAULT_TEXT_OVERLAP_THRESHOLD,
) -> Tuple[List[Record], DecontaminationReport]:
    """Flags a record as contaminated if either:
      - its image perceptual hash matches a held-out image exactly, or
      - the FRACTION of its own text n-grams that also appear in the held-out
        signature meets `text_overlap_threshold` (default 50%).

    Requiring a meaningful overlap fraction -- rather than flagging on any single
    shared 5-gram -- avoids false positives from generic boilerplate phrasing (e.g.
    templated uncertainty answers like "uncertain due to image quality...") that can
    legitimately recur across many unrelated training and eval examples. A genuine
    duplicate question+answer pair will have very high overlap and still gets caught.
    """
    kept: List[Record] = []
    report = DecontaminationReport()
    for r in records:
        report.total_checked += 1
        grams = text_ngrams(f"{r.question} {r.final_answer}")
        overlap_fraction = len(grams & eval_text_grams) / len(grams) if grams else 0.0
        text_hit = overlap_fraction >= text_overlap_threshold
        hashes = image_hashes(r.images)
        image_hit = bool(hashes & eval_image_hashes)

        if text_hit:
            report.flagged_text_overlap += 1
            report.flagged_ids.append(r.id)
            continue
        if image_hit:
            report.flagged_image_overlap += 1
            report.flagged_ids.append(r.id)
            continue
        kept.append(r)
    return kept, report
