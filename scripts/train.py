#!/usr/bin/env python
"""CLI: trains a MedLook-4B ablation, or validates config wiring locally with --dry-run.

Usage (local, no GPU, no model download):
    python scripts/train.py --config configs/smoke.yaml --dry-run

Usage (Colab, real training):
    python scripts/train.py --config configs/full_medlook.yaml

Usage (Colab, before committing to a full run -- catches multi-image packing failures
in under a minute on ~50 real samples):
    python scripts/train.py --config configs/full_medlook.yaml --packing-smoke-test 50

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medlook.train.sft import run_packing_smoke_test, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/full_medlook.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + count prepared samples only. No model download, no GPU required.",
    )
    parser.add_argument(
        "--packing-smoke-test",
        type=int,
        default=None,
        metavar="N",
        help="Run the real collator on the first N prepared samples and exit, without training. Requires a GPU.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to a checkpoint dir to resume from")
    args = parser.parse_args()

    if args.packing_smoke_test is not None:
        result = run_packing_smoke_test(args.config, num_samples=args.packing_smoke_test)
    else:
        result = run_training(
            args.config, dry_run=args.dry_run, resume_from_checkpoint=args.resume_from_checkpoint
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
