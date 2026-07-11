"""Curriculum mixing: combines Records from multiple adapters at configurable weights
and produces a stratified train/val split (stratified by adapter family + ACTION, so
neither split loses an entire class by chance).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from medlook.data.adapters.base import Record

DEFAULT_WEIGHTS = {
    "open_vqa": 0.475,
    "meissa": 0.325,
    "uncertainty": 0.2,
}


@dataclass
class CurriculumConfig:
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    val_fraction: float = 0.1
    seed: int = 3407


def adapter_family_for_record(record: Record) -> str:
    """`source` strings are prefixed by adapter family, e.g. "uncertainty_blur_vqa_rad"
    or "meissa_pathvqa", or a bare "vqa_rad"/"pathvqa"/"slake" from open_vqa."""
    if record.source.startswith("uncertainty"):
        return "uncertainty"
    if record.source.startswith("meissa"):
        return "meissa"
    return "open_vqa"


def mix_records(
    records_by_adapter: Dict[str, List[Record]],
    config: CurriculumConfig,
    target_size: Optional[int] = None,
) -> List[Record]:
    """Weighted sample from each adapter's record pool.

    If an adapter is unavailable (empty list -- e.g. "Meissa-off" mode), its weight is
    redistributed proportionally across the remaining available adapters rather than
    silently shrinking the dataset.
    """
    rng = random.Random(config.seed)

    available = {name: recs for name, recs in records_by_adapter.items() if recs}
    if not available:
        return []

    available_weight_total = sum(config.weights.get(name, 0.0) for name in available)
    if available_weight_total <= 0:
        normalized = {name: 1.0 / len(available) for name in available}
    else:
        normalized = {
            name: config.weights.get(name, 0.0) / available_weight_total for name in available
        }

    if target_size is None:
        target_size = sum(len(recs) for recs in available.values())

    mixed: List[Record] = []
    for name, recs in available.items():
        n = round(normalized[name] * target_size)
        pool = recs[:]
        rng.shuffle(pool)
        if n <= len(pool):
            mixed.extend(pool[:n])
        else:
            mixed.extend(pool)
            if pool:
                mixed.extend(rng.choices(pool, k=n - len(pool)))

    rng.shuffle(mixed)
    return mixed


def _strata_key(record: Record) -> Tuple[str, str]:
    action = record.strategy.action.value if record.strategy else "none"
    return (adapter_family_for_record(record), action)


def stratified_split(
    records: List[Record], val_fraction: float = 0.1, seed: int = 3407
) -> Tuple[List[Record], List[Record]]:
    """Splits records into train/val, preserving the (adapter_family, action)
    distribution in both splits as closely as possible."""
    rng = random.Random(seed)
    strata: Dict[Tuple[str, str], List[Record]] = {}
    for r in records:
        strata.setdefault(_strata_key(r), []).append(r)

    train: List[Record] = []
    val: List[Record] = []
    for group in strata.values():
        pool = group[:]
        rng.shuffle(pool)
        n_val = max(1, round(len(pool) * val_fraction)) if len(pool) > 1 else 0
        val.extend(pool[:n_val])
        train.extend(pool[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def action_histogram(records: List[Record]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in records:
        action = r.strategy.action.value if r.strategy else "none"
        hist[action] = hist.get(action, 0) + 1
    return hist


def image_count_histogram(records: List[Record]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in records:
        key = str(len(r.images))
        hist[key] = hist.get(key, 0) + 1
    return hist


def source_histogram(records: List[Record]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in records:
        hist[r.source] = hist.get(r.source, 0) + 1
    return hist
