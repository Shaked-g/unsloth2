"""Held-out gold strategy evaluation set.

This is the ONLY place strategy-selection accuracy/F1 should ever be measured against
(see `medlook/eval/strategy.py`). It is hand-reviewed, carved out before curriculum
mixing, and never eligible for training -- see PROJECT_BLUEPRINT.md Section 6 for why
this separation matters: grading against the same heuristics that generated training
labels (`strategy_labeler.py`) would be circular and would make the strategy-selection
metric meaningless.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

from PIL import Image

from medlook.schema import Action

LABELING_INSTRUCTIONS = """\
MedLook Gold Strategy Set -- Labeling Instructions
===================================================

This set exists to measure whether the model chooses the RIGHT visual strategy, not
just whether it answers correctly. It must be hand-reviewed and kept independent of any
automatic labeling heuristic used elsewhere in the pipeline (never generate it with
strategy_labeler.py).

For each case, assign exactly one gold_action:

  ANSWER_CONFIDENT -- The question is answerable directly and unambiguously from the
                       image(s) as given. A careful reviewer would not need a second
                       look.
  RELOOK           -- The question requires comparing regions, counting, or otherwise
                       benefits from an explicit second look / zoomed view, but IS
                       answerable given enough visual effort.
  FLAG_UNCERTAIN    -- Image quality, framing, or the inherent ambiguity of the finding
                       means a confident answer cannot honestly be given, with no
                       active contradiction to resolve.
  ESCALATE          -- There is a conflict (e.g. between image and accompanying text)
                       or a severity/ambiguity level that warrants human review rather
                       than an automated answer.

Target at least 150 cases, with FLAG_UNCERTAIN and ESCALATE deliberately
over-represented relative to their natural frequency in the training mix (they are the
rarest and most important classes to measure well). Every case needs a gold_answer
(the best honest answer/response for that action) and short notes explaining the
labeling rationale, so disagreements can be audited later.
"""


@dataclass
class GoldStrategyCase:
    id: str
    images: List[Image.Image]
    question: str
    gold_action: Action
    gold_answer: str
    notes: str = ""


def load_gold_strategy_set(json_path: str, image_dir: str) -> List[GoldStrategyCase]:
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    cases = []
    for entry in entries:
        images = [Image.open(os.path.join(image_dir, p)) for p in entry["images"]]
        cases.append(
            GoldStrategyCase(
                id=entry["id"],
                images=images,
                question=entry["question"],
                gold_action=Action(entry["gold_action"]),
                gold_answer=entry["gold_answer"],
                notes=entry.get("notes", ""),
            )
        )
    return cases


def action_distribution(cases: List[GoldStrategyCase]) -> Dict[str, int]:
    dist = {a.value: 0 for a in Action}
    for c in cases:
        dist[c.gold_action.value] += 1
    return dist


def print_labeling_instructions() -> None:
    print(LABELING_INSTRUCTIONS)
