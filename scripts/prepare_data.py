#!/usr/bin/env python
"""CLI: builds the MedLook training/val datasets from a YAML config.

Usage:
    python scripts/prepare_data.py --config configs/smoke.yaml

Always writes `data_stats.json` into the configured output directory, showing class
balance, per-adapter filter rejection counts, decontamination rate, and image-count
distribution -- so imbalance or quality problems are visible immediately rather than
discovered after a wasted Colab run.

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medlook.data import convert, curriculum, decontaminate, filter as data_filter, serialize
from medlook.data.adapters.base import Record
from medlook.data.adapters.meissa import MeissaAdapter
from medlook.data.adapters.open_vqa import OpenVQAAdapter
from medlook.data.adapters.uncertainty import UncertaintyAdapter
from medlook.data.gold_strategy_set import action_distribution, load_gold_strategy_set


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_open_vqa_adapter(cfg: dict) -> OpenVQAAdapter:
    mode = cfg.get("mode", "fixture")
    if mode == "fixture":
        return OpenVQAAdapter.from_fixture(cfg["fixture_path"], cfg["image_dir"])
    if mode == "huggingface":
        return OpenVQAAdapter.from_huggingface(
            sources=cfg.get("hf_sources", ("vqa_rad", "pathvqa", "slake")),
            split=cfg.get("hf_split", "train"),
            limit=cfg.get("hf_limit"),
        )
    raise ValueError(f"Unknown open_vqa mode {mode!r}")


def _build_meissa_adapter(cfg: dict) -> MeissaAdapter:
    mode = cfg.get("mode", "fixture")
    if mode == "fixture":
        return MeissaAdapter.from_fixture(cfg["fixture_path"], cfg["image_dir"])
    if mode == "huggingface":
        return MeissaAdapter.from_huggingface(
            split=cfg.get("hf_split", "train"), limit=cfg.get("hf_limit")
        )
    raise ValueError(f"Unknown meissa mode {mode!r}")


def _summarize_source_counts(records: List[Record]) -> Dict[str, int]:
    return curriculum.source_histogram(records)


def run(config_path: str, output_dir_override: str = None) -> dict:
    cfg = _load_config(config_path)
    data_cfg = cfg["data"]
    profile = cfg.get("profile", "full_medlook")
    output_dir = output_dir_override or data_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    stats: dict = {"config": config_path, "profile": profile, "adapters": {}}

    # --- open_vqa (primary, always required to be available) ---
    print("[prepare] stage: open_vqa", flush=True)
    open_vqa_adapter = _build_open_vqa_adapter(data_cfg["open_vqa"])
    open_vqa_available = open_vqa_adapter.is_available()
    if not open_vqa_available:
        print(
            "WARNING: open_vqa adapter produced zero records. This is the primary "
            "data source -- check the fixture/HF configuration.",
            file=sys.stderr,
        )
    open_vqa_records = list(open_vqa_adapter.iter_records())
    open_vqa_filtered, open_vqa_filter_stats = data_filter.apply_quality_filters(open_vqa_records)
    print(f"[prepare] open_vqa: {len(open_vqa_records)} raw -> {len(open_vqa_filtered)} filtered", flush=True)
    # Drop the unfiltered list; images are large and we only need the filtered pool.
    del open_vqa_records
    import gc

    gc.collect()
    stats["adapters"]["open_vqa"] = {
        "available": open_vqa_available,
        "raw_count": open_vqa_filter_stats.total_in,
        "filter": open_vqa_filter_stats.to_dict(),
    }

    # --- meissa (optional, graceful fallback) ---
    meissa_cfg = data_cfg.get("meissa", {})
    meissa_enabled = meissa_cfg.get("enabled", True)
    meissa_filtered: List[Record] = []
    if meissa_enabled:
        print("[prepare] stage: meissa", flush=True)
        meissa_adapter = _build_meissa_adapter(meissa_cfg)
        meissa_available = meissa_adapter.is_available()
        if not meissa_available:
            print(
                "INFO: meissa adapter unavailable (fixture missing, download failed, or "
                "empty after filtering). Continuing in Meissa-off mode using open_vqa + "
                "uncertainty alone.",
                file=sys.stderr,
            )
        meissa_records = list(meissa_adapter.iter_records())
        meissa_filtered, meissa_filter_stats = data_filter.apply_quality_filters(meissa_records)
        print(f"[prepare] meissa: {len(meissa_records)} raw -> {len(meissa_filtered)} filtered", flush=True)
        stats["adapters"]["meissa"] = {
            "available": meissa_available,
            "raw_count": len(meissa_records),
            "filter": meissa_filter_stats.to_dict(),
        }
    else:
        print("[prepare] stage: meissa (disabled)", flush=True)
        stats["adapters"]["meissa"] = {"available": False, "raw_count": 0, "filter": None}

    # --- uncertainty (anti-shortcut, derived from open_vqa's filtered records) ---
    unc_cfg = data_cfg.get("uncertainty", {})
    unc_enabled = unc_cfg.get("enabled", True)
    uncertainty_filtered: List[Record] = []
    if unc_enabled:
        print("[prepare] stage: uncertainty", flush=True)
        uncertainty_adapter = UncertaintyAdapter.from_open_vqa_records(
            open_vqa_filtered,
            blur_radius=unc_cfg.get("blur_radius", 6.0),
            crop_fraction=unc_cfg.get("crop_fraction", 0.5),
            max_bases=unc_cfg.get("max_bases", 750),
            seed=data_cfg.get("curriculum", {}).get("seed", 3407),
            max_image_side=unc_cfg.get("max_image_side", 512),
        )
        uncertainty_records = list(uncertainty_adapter.iter_records())
        uncertainty_filtered, uncertainty_filter_stats = data_filter.apply_quality_filters(
            uncertainty_records
        )
        print(
            f"[prepare] uncertainty: {len(uncertainty_records)} raw -> {len(uncertainty_filtered)} filtered",
            flush=True,
        )
        stats["adapters"]["uncertainty"] = {
            "available": uncertainty_adapter.is_available(),
            "raw_count": len(uncertainty_records),
            "filter": uncertainty_filter_stats.to_dict(),
        }
    else:
        print("[prepare] stage: uncertainty (disabled)", flush=True)
        stats["adapters"]["uncertainty"] = {"available": False, "raw_count": 0, "filter": None}

    # --- gold strategy set: held out, used ONLY to build the decontamination
    #     signature here; it is never mixed into training data. ---
    gold_cfg = data_cfg.get("gold_strategy_set")
    eval_text_grams, eval_image_hashes = set(), set()
    if gold_cfg:
        print("[prepare] stage: gold strategy set / decontam signature", flush=True)
        gold_cases = load_gold_strategy_set(gold_cfg["json_path"], gold_cfg["image_dir"])
        texts = [f"{c.question} {c.gold_answer}" for c in gold_cases]
        images_list = [c.images for c in gold_cases]
        eval_text_grams, eval_image_hashes = decontaminate.build_eval_signature_from_texts_images(
            texts, images_list
        )
        stats["gold_strategy_set"] = {
            "num_cases": len(gold_cases),
            "action_distribution": action_distribution(gold_cases),
        }
        print(f"[prepare] gold cases: {len(gold_cases)}", flush=True)

    # --- decontamination against the gold set (and, in the real pipeline, held-out
    #     eval subsets of VQA-RAD/SLAKE/PathVQA) ---
    print("[prepare] stage: decontamination", flush=True)
    records_by_adapter_raw = {
        "open_vqa": open_vqa_filtered,
        "meissa": meissa_filtered,
        "uncertainty": uncertainty_filtered,
    }
    records_by_adapter: Dict[str, List[Record]] = {}
    stats["decontamination"] = {}
    for name, recs in records_by_adapter_raw.items():
        if eval_text_grams or eval_image_hashes:
            kept, report = decontaminate.filter_contaminated(
                recs, eval_text_grams, eval_image_hashes
            )
        else:
            kept, report = recs, decontaminate.DecontaminationReport(total_checked=len(recs))
        records_by_adapter[name] = kept
        stats["decontamination"][name] = report.to_dict()
        print(f"[prepare] decontam {name}: {len(recs)} -> {len(kept)}", flush=True)

    # --- curriculum mixing + stratified split ---
    print("[prepare] stage: curriculum mix + split", flush=True)
    curriculum_cfg = data_cfg.get("curriculum", {})
    config = curriculum.CurriculumConfig(
        weights=curriculum_cfg.get("weights", dict(curriculum.DEFAULT_WEIGHTS)),
        val_fraction=curriculum_cfg.get("val_fraction", 0.1),
        seed=curriculum_cfg.get("seed", 3407),
    )
    target_size = curriculum_cfg.get("target_size")
    mixed = curriculum.mix_records(records_by_adapter, config, target_size=target_size)
    train_records, val_records = curriculum.stratified_split(
        mixed, val_fraction=config.val_fraction, seed=config.seed
    )

    train_action_hist = curriculum.action_histogram(train_records)
    val_action_hist = curriculum.action_histogram(val_records)
    balance_warnings = data_filter.check_class_balance(train_action_hist)
    for w in balance_warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    stats["curriculum"] = {
        "weights_configured": config.weights,
        "val_fraction": config.val_fraction,
        "seed": config.seed,
        "target_size": target_size,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "train_action_histogram": train_action_hist,
        "val_action_histogram": val_action_hist,
        "train_image_count_histogram": curriculum.image_count_histogram(train_records),
        "train_source_histogram": _summarize_source_counts(train_records),
        "class_balance_warnings": balance_warnings,
    }

    # --- convert + serialize ---
    print("[prepare] stage: convert + serialize", flush=True)
    train_samples = convert.records_to_unsloth_dataset(train_records, profile=profile)
    val_samples = convert.records_to_unsloth_dataset(val_records, profile=profile)
    train_path = serialize.save_dataset_jsonl(train_samples, output_dir, "train")
    val_path = serialize.save_dataset_jsonl(val_samples, output_dir, "val")
    stats["output"] = {"train_path": train_path, "val_path": val_path, "output_dir": output_dir}

    stats_path = os.path.join(output_dir, "data_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {len(train_records)} train / {len(val_records)} val samples to {output_dir}")
    print(f"data_stats.json: {stats_path}")
    print(f"Train action histogram: {train_action_hist}")
    if balance_warnings:
        print(f"Class balance warnings: {balance_warnings}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/smoke.yaml")
    parser.add_argument(
        "--output-dir", default=None, help="Override the config's data.output_dir"
    )
    args = parser.parse_args()
    run(args.config, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
