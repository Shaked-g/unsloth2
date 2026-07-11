#!/usr/bin/env python
"""CLI: runs REAL inference for one system (base / short_sft / process_sft /
full_medlook) over held-out answer-quality sets and/or the gold strategy set, and
writes raw generations into the on-disk layout `scripts/eval.py --predictions-dir`
expects (see `medlook.eval.runner.load_predictions_from_dir`).

This is the piece that turns a trained adapter into scoreable predictions -- run it
once per system on Colab, then run `scripts/eval.py --predictions-dir` once against
the combined output directory to get the four-system report.

IMPORTANT: use a `--split` that was NOT used for training (see `scripts/prepare_data.py`,
which defaults to `hf_split: train`) -- e.g. "test" or "validation" depending on what
each underlying HF dataset actually provides. Verify the real split names for
vqa_rad/pathvqa/slake before relying on this; this script does not invent a held-out
split if the requested one does not exist upstream, it will simply fail loudly.

Usage:
    # Base system: no adapter.
    python scripts/generate_predictions.py --config configs/full_medlook.yaml \\
        --system-name base --split test --out-dir predictions/

    # A trained ablation: pass its adapter.
    python scripts/generate_predictions.py --config configs/full_medlook.yaml \\
        --adapter-dir /content/drive/MyDrive/medlook_runs/full_medlook/final_adapter \\
        --system-name full_medlook --split test --out-dir predictions/ \\
        --gold-strategy-json data/gold_strategy_set.json --gold-strategy-image-dir data/gold_strategy_set_images

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions. Requires unsloth + torch + a GPU; never run this locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medlook.data.adapters.open_vqa import OpenVQAAdapter
from medlook.data.gold_strategy_set import load_gold_strategy_set
from medlook.inference import generate_with_model
from medlook.train.sft import load_model_with_optional_adapter


def _write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_answer_quality_predictions(
    model, tokenizer, sources, split: str, limit, system_name: str, out_dir: str, max_new_tokens: int
) -> None:
    for source in sources:
        adapter = OpenVQAAdapter.from_huggingface(sources=[source], split=split, limit=limit)
        if not adapter.is_available():
            print(f"WARNING: no records for source={source!r} split={split!r}; skipping", file=sys.stderr)
            continue
        rows = []
        for record in adapter.iter_records():
            raw = generate_with_model(model, tokenizer, record.images, record.question, max_new_tokens=max_new_tokens)
            rows.append(
                {
                    "id": record.id,
                    "question": record.question,
                    "gold_answer": record.final_answer,
                    "raw_generation": raw,
                }
            )
        out_path = os.path.join(out_dir, "answer_quality", system_name, f"{source}.jsonl")
        _write_jsonl(out_path, rows)
        print(f"Wrote {len(rows)} predictions to {out_path}")


def generate_gold_strategy_predictions(
    model, tokenizer, json_path: str, image_dir: str, system_name: str, out_dir: str, max_new_tokens: int
) -> None:
    cases = load_gold_strategy_set(json_path, image_dir)
    rows = []
    for case in cases:
        raw = generate_with_model(model, tokenizer, case.images, case.question, max_new_tokens=max_new_tokens)
        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "gold_answer": case.gold_answer,
                "raw_generation": raw,
                "gold_action": case.gold_action.value,
            }
        )
    out_path = os.path.join(out_dir, "gold_strategy", f"{system_name}.jsonl")
    _write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} predictions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config identifying the base model")
    parser.add_argument("--adapter-dir", default=None, help="Trained LoRA adapter dir (omit for the Base system)")
    parser.add_argument("--system-name", required=True, choices=["base", "short_sft", "process_sft", "full_medlook"])
    parser.add_argument("--out-dir", required=True, help="Root predictions directory (see module docstring)")
    parser.add_argument("--sources", default="vqa_rad,pathvqa,slake", help="Comma-separated open_vqa sources")
    parser.add_argument("--split", default="test", help="HF split to use -- MUST differ from the training split")
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per source (useful for a quick pass)")
    parser.add_argument("--gold-strategy-json", default=None, help="Path to the gold strategy set JSON (skip if omitted)")
    parser.add_argument("--gold-strategy-image-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_model_with_optional_adapter(args.config, args.adapter_dir)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    generate_answer_quality_predictions(
        model, tokenizer, sources, args.split, args.limit, args.system_name, args.out_dir, args.max_new_tokens
    )

    if args.gold_strategy_json:
        generate_gold_strategy_predictions(
            model,
            tokenizer,
            args.gold_strategy_json,
            args.gold_strategy_image_dir,
            args.system_name,
            args.out_dir,
            args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
