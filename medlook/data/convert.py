"""Converts common intermediate `Record` objects into the exact message format that
Unsloth's `FastVisionModel` / `UnslothVisionDataCollator` expect:

    {"messages": [
        {"role": "user", "content": [{"type": "text", "text": question},
                                      {"type": "image", "image": pil_image}, ...]},
        {"role": "assistant", "content": [{"type": "text", "text": rendered_schema}]}
    ]}

The rendered assistant text always goes through `medlook.schema.render_and_validate`,
so a malformed Record fails loudly here rather than silently producing a bad training
target.

`profile` selects which schema blocks are emitted, matching the three ablations:
  - "short_sft":    [FINAL] only
  - "process_sft":  [PROCESS] + [FINAL], no [STRATEGY]
  - "full_medlook": [STRATEGY] + optional [PROCESS] + [FINAL]
"""

from __future__ import annotations

from typing import Iterable, List

from medlook.data.adapters.base import Record
from medlook.schema import MedLookResponse, render_and_validate

PROFILES = ("short_sft", "process_sft", "full_medlook")


def _build_response(record: Record, profile: str) -> MedLookResponse:
    if profile == "short_sft":
        return MedLookResponse(
            final_answer=record.final_answer, final_confidence=record.final_confidence
        )
    if profile == "process_sft":
        return MedLookResponse(
            final_answer=record.final_answer,
            final_confidence=record.final_confidence,
            process=record.process,
        )
    if profile == "full_medlook":
        return MedLookResponse(
            final_answer=record.final_answer,
            final_confidence=record.final_confidence,
            strategy=record.strategy,
            process=record.process,
        )
    raise ValueError(f"Unknown profile {profile!r}. Must be one of {PROFILES}.")


def record_to_unsloth_sample(record: Record, profile: str = "full_medlook") -> dict:
    response = _build_response(record, profile)
    rendered = render_and_validate(response)

    user_content = [{"type": "text", "text": record.question}]
    for image in record.images:
        user_content.append({"type": "image", "image": image})

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": rendered}]},
        ],
        "meta": {
            "id": record.id,
            "source": record.source,
            "difficulty": record.difficulty,
            "action": record.strategy.action.value if record.strategy else None,
            "num_images": len(record.images),
            "gold": record.gold,
        },
    }


def records_to_unsloth_dataset(
    records: Iterable[Record], profile: str = "full_medlook"
) -> List[dict]:
    """List-comprehension style conversion, preferred over a heavy `.map` call for
    multi-image stability (per Unsloth's own guidance for vision datasets)."""
    return [record_to_unsloth_sample(r, profile=profile) for r in records]
