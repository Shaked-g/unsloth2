"""`open_vqa` adapter -- the PRIMARY, always-available data source.

Converts open, freely-licensed medical VQA datasets (PathVQA, SLAKE, VQA-RAD) into
common `Record` objects. Unlike the optional `meissa` adapter, this requires no
credentialing and no dataset-specific license friction, so `prepare_data.py` can always
build a working dataset from this adapter alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from PIL import Image

from medlook.data.adapters.base import Adapter, Record
from medlook.data.strategy_labeler import label_question_answer
from medlook.schema import Action, Strategy

# Open, non-credentialed Hugging Face mirrors used for the real (Colab) run.
HF_SOURCES = {
    "vqa_rad": "flaviagiammarino/vqa-rad",
    "pathvqa": "flaviagiammarino/path-vqa",
    "slake": "mdwiratathya/SLAKE-vqa-english",
}

_CONFIDENCE_BY_ACTION = {
    Action.ANSWER_CONFIDENT: 0.9,
    Action.RELOOK: 0.75,
    Action.FLAG_UNCERTAIN: 0.4,
    Action.ESCALATE: 0.3,
}


@dataclass
class RawVQARecord:
    id: str
    source: str
    image: Image.Image
    question: str
    answer: str


class OpenVQAAdapter(Adapter):
    name = "open_vqa"

    def __init__(self, raw_records: Iterable[RawVQARecord]):
        self._raw_records: List[RawVQARecord] = list(raw_records)

    def is_available(self) -> bool:
        return len(self._raw_records) > 0

    def iter_records(self) -> Iterator[Record]:
        for raw in self._raw_records:
            label = label_question_answer(raw.question, raw.answer)
            strategy = Strategy(action=label.action, conf=label.conf, reason=label.reason)
            process = _process_for_relook(raw.question) if label.action == Action.RELOOK else None
            yield Record(
                id=raw.id,
                images=[raw.image if getattr(raw.image, "mode", None) == "RGB" else _rgb_maybe_resize(raw.image)],
                question=raw.question,
                strategy=strategy,
                final_answer=raw.answer,
                final_confidence=_CONFIDENCE_BY_ACTION.get(label.action, 0.5),
                process=process,
                source=raw.source,
                difficulty="easy" if label.action == Action.ANSWER_CONFIDENT else "medium",
                extra={},
            )

    @classmethod
    def from_fixture(cls, fixture_path: str, image_dir: str) -> "OpenVQAAdapter":
        """Load raw records from a local JSON fixture. Never requires network access;
        used for local tests and the `smoke` config."""
        with open(fixture_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        raw_records = []
        for entry in entries:
            image_path = os.path.join(image_dir, entry["image"])
            image = Image.open(image_path)
            raw_records.append(
                RawVQARecord(
                    id=entry["id"],
                    source=entry["source"],
                    image=image,
                    question=entry["question"],
                    answer=entry["answer"],
                )
            )
        return cls(raw_records)

    @classmethod
    def from_huggingface(
        cls,
        sources: Iterable[str] = ("vqa_rad", "pathvqa", "slake"),
        split: str = "train",
        limit: Optional[int] = None,
    ) -> "OpenVQAAdapter":
        """Load raw records from the open Hugging Face mirrors of PathVQA, SLAKE, and
        VQA-RAD. Requires network access and the `datasets` package -- used by the
        real Colab data-preparation run, never by local tests. On any failure this
        returns an empty (unavailable) adapter rather than raising, since `open_vqa`
        being unreachable should be visible in `data_stats.json`, not a hard crash."""
        try:
            import datasets  # local import: keeps this heavy dependency optional for tests
        except ImportError:
            return cls([])

        raw_records: List[RawVQARecord] = []
        for source in sources:
            hf_id = HF_SOURCES[source]
            print(f"[open_vqa] Loading {hf_id} split={split!r} ...", flush=True)
            try:
                ds = datasets.load_dataset(hf_id, split=split)
            except Exception as exc:
                print(f"[open_vqa] FAILED {source}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if limit is not None:
                ds = ds.select(range(min(limit, len(ds))))
            print(f"[open_vqa] {source}: taking {len(ds)} examples", flush=True)
            for i, row in enumerate(ds):
                raw_records.append(
                    RawVQARecord(
                        id=f"{source}_{i}",
                        source=source,
                        image=_rgb_maybe_resize(row["image"]),
                        question=row["question"],
                        answer=str(row["answer"]),
                    )
                )
        print(f"[open_vqa] total raw records: {len(raw_records)}", flush=True)
        return cls(raw_records)


def _rgb_maybe_resize(image: Image.Image, max_side: int = 512) -> Image.Image:
    """Keep training images RGB and memory-bounded for Colab/laptop prep."""
    out = image.convert("RGB")
    w, h = out.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        out = out.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return out


def _process_for_relook(question: str) -> str:
    return f'Re-examined the relevant region(s) of the image before answering: "{question.strip()}"'
