"""`meissa` adapter -- OPTIONAL enrichment source.

Provides real multi-step, multi-image RELOOK trajectories distilled from Meissa's
`interleaved_thinking_images` framework (`CYX1998/Meissa-SFT`, Apache-2.0, ~10.5k
samples across PathVQA/SLAKE/VQA-RAD). See the Meissa paper (arXiv:2603.09018) and
dataset card for details.

This adapter is OPTIONAL. If the dataset cannot be downloaded (network issue, HF
outage, or the operator simply chooses not to use it), `is_available()` returns False
and `prepare_data.py` continues in "Meissa-off" mode using `open_vqa` + `uncertainty`
alone. The pipeline must never hard-fail because this adapter is unavailable.

Tool-call steps map to a real RELOOK trajectory: when a genuine second (cropped) image
is available in the sample it is kept as a real second image, never fabricated. Zero-
tool finals map to ANSWER_CONFIDENT.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from PIL import Image

from medlook.data.adapters.base import Adapter, Record
from medlook.schema import Action, Strategy

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FINAL_TAG_RE = re.compile(r"\[FINAL\]\s*(.*)", re.DOTALL)
_IMAGE_TAG_RE = re.compile(r"<image>\s*")


@dataclass
class RawMeissaSample:
    id: str
    source_dataset: str
    conversations: List[dict]
    images: List[Image.Image]


class MeissaAdapter(Adapter):
    name = "meissa"

    def __init__(self, raw_samples: Iterable[RawMeissaSample]):
        self._raw_samples: List[RawMeissaSample] = list(raw_samples)

    def is_available(self) -> bool:
        return len(self._raw_samples) > 0

    def iter_records(self) -> Iterator[Record]:
        for raw in self._raw_samples:
            record = _convert_sample(raw)
            if record is not None:
                yield record

    @classmethod
    def from_fixture(cls, fixture_path: str, image_dir: str) -> "MeissaAdapter":
        """Load raw samples from a local ShareGPT-style JSON fixture mirroring the
        real Meissa-SFT format. Returns an empty (unavailable) adapter if the fixture
        is missing or malformed, exercising the graceful-fallback path in tests."""
        try:
            with open(fixture_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls([])

        raw_samples = []
        for entry in entries:
            try:
                images = [Image.open(os.path.join(image_dir, p)) for p in entry["images"]]
            except FileNotFoundError:
                continue
            meta = entry.get("meta", {})
            raw_samples.append(
                RawMeissaSample(
                    id=str(meta.get("idx", meta.get("dataset", "meissa"))),
                    source_dataset=meta.get("dataset", "meissa"),
                    conversations=entry["conversations"],
                    images=images,
                )
            )
        return cls(raw_samples)

    @classmethod
    def from_huggingface(
        cls, split: str = "train", limit: Optional[int] = None
    ) -> "MeissaAdapter":
        """Attempt to download `CYX1998/Meissa-SFT` and filter to the interleaved-
        images framework. Returns an empty (unavailable) adapter on ANY failure --
        this source is optional enrichment, never a hard dependency.

        Scans row-by-row (with early stop at `limit`) instead of a full-dataset
        `.filter(...)` pass -- the latter can hang for a long time on large Hub
        datasets with no progress output.
        """
        try:
            import datasets

            print(f"[meissa] Loading CYX1998/Meissa-SFT split={split!r} ...", flush=True)
            ds = datasets.load_dataset("CYX1998/Meissa-SFT", split=split)
            print(f"[meissa] Loaded {len(ds)} raw rows; scanning for interleaved_thinking_images ...", flush=True)
        except Exception as exc:
            print(f"[meissa] unavailable ({type(exc).__name__}: {exc})", flush=True)
            return cls([])

        raw_samples: List[RawMeissaSample] = []
        try:
            for i, row in enumerate(ds):
                meta = row.get("meta", {}) or {}
                if meta.get("framework") != "interleaved_thinking_images":
                    continue
                try:
                    images = list(row["images"])
                except Exception:
                    images = []
                if not images:
                    continue
                raw_samples.append(
                    RawMeissaSample(
                        id=str(meta.get("idx", f"meissa_{i}")),
                        source_dataset=meta.get("dataset", "meissa"),
                        conversations=row["conversations"],
                        images=images,
                    )
                )
                if limit is not None and len(raw_samples) >= limit:
                    break
                if len(raw_samples) % 200 == 0 and len(raw_samples) > 0:
                    print(f"[meissa] kept {len(raw_samples)} (scanned {i + 1}) ...", flush=True)
        except Exception as exc:
            print(f"[meissa] scan failed ({type(exc).__name__}: {exc}); returning {len(raw_samples)} kept", flush=True)

        print(f"[meissa] ready with {len(raw_samples)} samples", flush=True)
        return cls(raw_samples)


def _strip_think(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def _extract_final_answer(gpt_text: str) -> Optional[str]:
    cleaned = _strip_think(gpt_text)
    match = _FINAL_TAG_RE.search(cleaned)
    if match:
        return match.group(1).strip() or None
    return cleaned or None


def _convert_sample(raw: RawMeissaSample) -> Optional[Record]:
    conversations = raw.conversations
    if not conversations:
        return None

    human_turns = [c for c in conversations if c.get("from") == "human"]
    if not human_turns:
        return None
    question = _IMAGE_TAG_RE.sub("", human_turns[0]["value"]).strip()

    tool_steps = [c for c in conversations if c.get("from") == "function_call"]
    gpt_turns = [c for c in conversations if c.get("from") == "gpt"]
    if not gpt_turns:
        return None
    final_answer = _extract_final_answer(gpt_turns[-1]["value"])
    if not final_answer:
        return None

    if tool_steps:
        # Real multi-step trajectory: map each tool-call step's reasoning into an
        # explicit textual re-examination process. If real cropped images beyond the
        # original are present in raw.images, they are kept as genuine second/third
        # images rather than only described in text (see PROJECT_BLUEPRINT.md
        # Section 6, "real multi-image RELOOK").
        process_lines = []
        for step in tool_steps:
            reasoning = _strip_think(step.get("value", ""))
            process_lines.append(
                reasoning if reasoning else "Examined a specific image region more closely."
            )
        process = " ".join(process_lines)
        action = Action.RELOOK
        strategy_conf = 0.65
        reason = "Trajectory included tool-assisted re-examination before answering."
        images = raw.images[:3] if raw.images else []
        final_confidence = 0.7
        difficulty = "medium"
    else:
        process = None
        action = Action.ANSWER_CONFIDENT
        strategy_conf = 0.85
        reason = "Trajectory answered directly with no tool-assisted re-examination."
        images = raw.images[:1] if raw.images else []
        final_confidence = 0.8
        difficulty = "easy"

    if not images:
        return None

    return Record(
        id=raw.id,
        images=images,
        question=question,
        strategy=Strategy(action=action, conf=strategy_conf, reason=reason),
        final_answer=final_answer,
        final_confidence=final_confidence,
        process=process,
        source=f"meissa_{raw.source_dataset}",
        difficulty=difficulty,
        extra={"num_tool_steps": len(tool_steps)},
    )
