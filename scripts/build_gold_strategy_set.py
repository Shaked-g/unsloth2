#!/usr/bin/env python
"""Build a held-out gold strategy evaluation set from open VQA *test* splits.

This set is NEVER used for training. Labels are assigned by an independent curation
policy documented in GOLD_CURATION_POLICY below -- intentionally NOT by
`medlook.data.strategy_labeler` (that module is for training labels only; grading
against it would be circular).

Target composition (default ~200 cases):
  - ~50 ANSWER_CONFIDENT  -- clean, short closed-form Q/A from held-out test
  - ~50 RELOOK            -- counting / compare / multi-region questions
  - ~50 FLAG_UNCERTAIN    -- degraded counterparts of confident cases (anti-shortcut)
  - ~50 ESCALATE          -- conflicting clinical-note cases

Usage:
    python scripts/build_gold_strategy_set.py \\
        --out-json data/gold_strategy_set.json \\
        --out-image-dir data/gold_strategy_set_images \\
        --per-action 50

Disclaimer: research prototype only. Not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medlook.data.adapters.open_vqa import HF_SOURCES
from medlook.schema import Action

GOLD_CURATION_POLICY = """
Independent of strategy_labeler.py. Cases are taken only from HF *test* splits.

ANSWER_CONFIDENT:
  Closed-form question (starts with is/are/does/do/was/were/has/have) AND gold answer
  has <= 3 tokens AND answer is not hedge language. Image is the original clean test image.

RELOOK:
  Question contains counting/compare/multi-region cues (how many, compare, difference,
  both, each, which is larger, relationship, spatial) OR answer is a long (>=12 token)
  finding description. Clean test image.

FLAG_UNCERTAIN:
  Same question as a curated ANSWER_CONFIDENT case, but image is heavily Gaussian-blurred
  so the finding is no longer reliably answerable. Gold answer states uncertainty.
  Shares parent_id with the clean counterpart (anti-shortcut pairing for eval).

ESCALATE:
  Same image as a curated ANSWER_CONFIDENT case, but the question is rewritten with an
  explicit conflicting clinical note that contradicts the gold finding. Gold answer
  instructs escalation / human review.
"""

_CLOSED_STARTS = ("is ", "are ", "does ", "do ", "was ", "were ", "has ", "have ")
_HEDGE = (
    "uncertain",
    "unclear",
    "cannot",
    "can't",
    "not sure",
    "ambiguous",
    "inconclusive",
    "indeterminate",
)
_RELOOK_CUES = (
    "how many",
    "compare",
    "comparison",
    "difference between",
    "both",
    "each",
    "which is larger",
    "which is bigger",
    "relationship",
    "spatial",
    "relative",
)


def _word_count(text: str) -> int:
    return len(text.strip().split())


def _is_confident_candidate(question: str, answer: str) -> bool:
    q = question.strip().lower()
    a = answer.strip().lower()
    if any(h in a for h in _HEDGE):
        return False
    return q.startswith(_CLOSED_STARTS) and _word_count(answer) <= 3


def _is_relook_candidate(question: str, answer: str) -> bool:
    q = question.strip().lower()
    if any(c in q for c in _RELOOK_CUES):
        return True
    return _word_count(answer) >= 12


def _load_test_pool(limit_per_source: int | None = None) -> List[dict]:
    import datasets

    pool = []
    for source, hf_id in HF_SOURCES.items():
        try:
            ds = datasets.load_dataset(hf_id, split="test")
        except Exception as exc:
            print(f"WARNING: could not load test split for {source}: {exc}", file=sys.stderr)
            continue
        if limit_per_source is not None:
            ds = ds.select(range(min(limit_per_source, len(ds))))
        for i, row in enumerate(ds):
            pool.append(
                {
                    "id": f"{source}_test_{i}",
                    "source": source,
                    "image": row["image"],
                    "question": row["question"],
                    "answer": str(row["answer"]),
                }
            )
    return pool


def _save_image(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.convert("RGB").save(path)


def build_gold_set(per_action: int, seed: int, blur_radius: float = 8.0) -> Tuple[List[dict], Dict[str, Image.Image]]:
    rng = random.Random(seed)
    pool = _load_test_pool()
    if not pool:
        raise RuntimeError("No test-split examples loaded; cannot build gold set.")

    confident = [ex for ex in pool if _is_confident_candidate(ex["question"], ex["answer"])]
    relook = [ex for ex in pool if _is_relook_candidate(ex["question"], ex["answer"])]
    # Avoid double-counting: prefer pure relook that aren't also simple closed+short
    relook = [ex for ex in relook if not _is_confident_candidate(ex["question"], ex["answer"])]

    rng.shuffle(confident)
    rng.shuffle(relook)

    entries: List[dict] = []
    images: Dict[str, Image.Image] = {}

    confident_picked = confident[:per_action]
    relook_picked = relook[:per_action]

    for ex in confident_picked:
        fname = f"{ex['id']}_clean.png"
        images[fname] = ex["image"]
        entries.append(
            {
                "id": f"gold_conf_{ex['id']}",
                "images": [fname],
                "question": ex["question"],
                "gold_action": Action.ANSWER_CONFIDENT.value,
                "gold_answer": ex["answer"],
                "notes": (
                    "Curated ANSWER_CONFIDENT from held-out test: closed-form question, "
                    f"short gold answer. source={ex['source']}"
                ),
                "parent_id": ex["id"],
                "source": ex["source"],
            }
        )

    for ex in relook_picked:
        fname = f"{ex['id']}_clean.png"
        images[fname] = ex["image"]
        entries.append(
            {
                "id": f"gold_relook_{ex['id']}",
                "images": [fname],
                "question": ex["question"],
                "gold_action": Action.RELOOK.value,
                "gold_answer": ex["answer"],
                "notes": (
                    "Curated RELOOK from held-out test: counting/compare/long-finding. "
                    f"source={ex['source']}"
                ),
                "parent_id": ex["id"],
                "source": ex["source"],
            }
        )

    # FLAG / ESCALATE from confident parents (anti-shortcut + conflict)
    flag_parents = confident_picked[:per_action]
    for ex in flag_parents:
        fname = f"{ex['id']}_blur.png"
        blurred = ex["image"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius))
        images[fname] = blurred
        entries.append(
            {
                "id": f"gold_flag_{ex['id']}",
                "images": [fname],
                "question": ex["question"],
                "gold_action": Action.FLAG_UNCERTAIN.value,
                "gold_answer": (
                    "Uncertain due to image quality; the finding cannot be confidently "
                    "confirmed or excluded from this degraded view."
                ),
                "notes": (
                    "Anti-shortcut FLAG: same question as gold_conf counterpart, heavily "
                    f"blurred image. parent={ex['id']}"
                ),
                "parent_id": ex["id"],
                "source": ex["source"],
            }
        )

    escalate_parents = confident_picked[:per_action]
    for ex in escalate_parents:
        fname = f"{ex['id']}_escalate.png"
        images[fname] = ex["image"]
        conflict_q = (
            f"{ex['question']} "
            f"[Accompanying note claims the opposite finding of '{ex['answer']}' "
            f"and requests automated confirmation.]"
        )
        entries.append(
            {
                "id": f"gold_esc_{ex['id']}",
                "images": [fname],
                "question": conflict_q,
                "gold_action": Action.ESCALATE.value,
                "gold_answer": (
                    "The accompanying note conflicts with what can be assessed from the "
                    "image; this should be escalated for human review rather than answered "
                    "directly."
                ),
                "notes": (
                    "ESCALATE: conflicting clinical note vs image/gold finding. "
                    f"parent={ex['id']}"
                ),
                "parent_id": ex["id"],
                "source": ex["source"],
            }
        )

    return entries, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-json", default="data/gold_strategy_set.json")
    parser.add_argument("--out-image-dir", default="data/gold_strategy_set_images")
    parser.add_argument("--per-action", type=int, default=50, help="Target cases per ACTION class")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--blur-radius", type=float, default=8.0)
    args = parser.parse_args()

    print(GOLD_CURATION_POLICY)
    entries, images = build_gold_set(args.per_action, args.seed, blur_radius=args.blur_radius)

    os.makedirs(args.out_image_dir, exist_ok=True)
    for fname, img in images.items():
        _save_image(img, os.path.join(args.out_image_dir, fname))

    # JSON only needs relative filenames (already relative)
    serializable = [
        {k: v for k, v in e.items() if k != "image"}
        for e in entries
    ]
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    dist: Dict[str, int] = {}
    for e in entries:
        dist[e["gold_action"]] = dist.get(e["gold_action"], 0) + 1
    print(f"Wrote {len(entries)} gold cases to {args.out_json}")
    print(f"Images: {len(images)} files in {args.out_image_dir}")
    print(f"Action distribution: {dist}")
    shortfalls = [a for a, n in dist.items() if n < args.per_action]
    if shortfalls:
        print(
            f"WARNING: fewer than --per-action={args.per_action} cases for {shortfalls}. "
            "Held-out test pools may be small; consider lowering --per-action.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
