"""`uncertainty` adapter -- purpose-built FLAG_UNCERTAIN / ESCALATE training signal.

Anti-shortcut design (see PROJECT_BLUEPRINT.md Section 6 / Section 4.3 of the original
hardening instructions): every degraded example is generated ALONGSIDE its clean
counterpart from the SAME base image and question. The clean version is labeled
ANSWER_CONFIDENT and the degraded versions FLAG_UNCERTAIN or ESCALATE. Because topic
and difficulty are held constant and only the perturbation differs between a pair, the
model cannot learn "blurry image texture" as a shortcut for the FLAG label divorced
from actual answerability -- the only reliable signal distinguishing a pair is whether
the question can still be confidently answered from the image.

Four variants are produced per clean base example:
  - clean:    unmodified image                       -> ANSWER_CONFIDENT
  - blur:     Gaussian blur                          -> FLAG_UNCERTAIN
  - crop:     aggressive crop-away of the field       -> FLAG_UNCERTAIN
  - conflict: mild blur + a conflicting text hint     -> ESCALATE
All four share an `extra["pair_id"]` linking them back to the same base example, so the
pairing can be verified/audited later (e.g. in `data_stats.json` or ad hoc analysis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List

from PIL import Image, ImageFilter

from medlook.data.adapters.base import Adapter, Record
from medlook.schema import Action, Strategy


@dataclass
class RawCleanExample:
    id: str
    image: Image.Image
    question: str
    answer: str
    source: str


class UncertaintyAdapter(Adapter):
    name = "uncertainty"

    def __init__(
        self,
        clean_examples: Iterable[RawCleanExample],
        blur_radius: float = 6.0,
        crop_fraction: float = 0.5,
    ):
        self._clean_examples: List[RawCleanExample] = list(clean_examples)
        self._blur_radius = blur_radius
        self._crop_fraction = crop_fraction

    def is_available(self) -> bool:
        return len(self._clean_examples) > 0

    def iter_records(self) -> Iterator[Record]:
        for ex in self._clean_examples:
            yield _make_clean_record(ex)
            yield _make_blurred_record(ex, self._blur_radius)
            yield _make_cropped_away_record(ex, self._crop_fraction)
            yield _make_conflicting_hint_record(ex, self._blur_radius)

    @classmethod
    def from_open_vqa_records(cls, records: Iterable, **kwargs) -> "UncertaintyAdapter":
        """Build clean examples directly from already-converted `open_vqa` Records, so
        the SAME base images/questions get their degraded counterparts -- this is what
        makes the pairing meaningful rather than coincidental."""
        clean_examples = []
        for r in records:
            if not r.images:
                continue
            clean_examples.append(
                RawCleanExample(
                    id=r.id,
                    image=r.images[0],
                    question=r.question,
                    answer=r.final_answer,
                    source=r.source,
                )
            )
        return cls(clean_examples, **kwargs)


def _make_clean_record(ex: RawCleanExample) -> Record:
    return Record(
        id=f"{ex.id}_unc_clean",
        images=[ex.image],
        question=ex.question,
        strategy=Strategy(
            action=Action.ANSWER_CONFIDENT,
            conf=0.88,
            reason="Image is clear; the finding is directly visible.",
        ),
        final_answer=ex.answer,
        final_confidence=0.88,
        process=None,
        source=f"uncertainty_clean_{ex.source}",
        difficulty="easy",
        extra={"pair_id": ex.id, "variant": "clean"},
    )


def _make_blurred_record(ex: RawCleanExample, blur_radius: float) -> Record:
    degraded = ex.image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Record(
        id=f"{ex.id}_unc_blur",
        images=[degraded],
        question=ex.question,
        strategy=Strategy(
            action=Action.FLAG_UNCERTAIN,
            conf=0.35,
            reason="Image is too blurred to confidently confirm or exclude the finding.",
        ),
        final_answer=(
            "Uncertain due to image quality; cannot confidently confirm or exclude the finding."
        ),
        final_confidence=0.35,
        process=None,
        source=f"uncertainty_blur_{ex.source}",
        difficulty="hard",
        extra={"pair_id": ex.id, "variant": "blur"},
    )


def _make_cropped_away_record(ex: RawCleanExample, crop_fraction: float) -> Record:
    w, h = ex.image.size
    cw, ch = max(int(w * crop_fraction), 1), max(int(h * crop_fraction), 1)
    degraded = ex.image.crop((0, 0, cw, ch)).resize((w, h))
    return Record(
        id=f"{ex.id}_unc_crop",
        images=[degraded],
        question=ex.question,
        strategy=Strategy(
            action=Action.FLAG_UNCERTAIN,
            conf=0.3,
            reason="The relevant region may have been cropped out of the visible field.",
        ),
        final_answer=(
            "Uncertain; the region needed to answer this question may not be fully visible."
        ),
        final_confidence=0.3,
        process=None,
        source=f"uncertainty_crop_{ex.source}",
        difficulty="hard",
        extra={"pair_id": ex.id, "variant": "crop"},
    )


def _make_conflicting_hint_record(ex: RawCleanExample, blur_radius: float) -> Record:
    degraded = ex.image.filter(ImageFilter.GaussianBlur(radius=max(blur_radius / 2, 1.0)))
    conflicting_question = (
        f"{ex.question} (Note: the accompanying report states a different, "
        f"inconsistent finding than what may be visible here.)"
    )
    return Record(
        id=f"{ex.id}_unc_conflict",
        images=[degraded],
        question=conflicting_question,
        strategy=Strategy(
            action=Action.ESCALATE,
            conf=0.25,
            reason="Visual evidence and the accompanying text hint conflict; needs human review.",
        ),
        final_answer=(
            "The image and the accompanying note appear inconsistent; this should be "
            "escalated for human review."
        ),
        final_confidence=0.25,
        process=None,
        source=f"uncertainty_conflict_{ex.source}",
        difficulty="hard",
        extra={"pair_id": ex.id, "variant": "conflict"},
    )
