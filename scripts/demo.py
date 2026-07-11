#!/usr/bin/env python
"""CLI: launches the MedLook-4B Gradio demo.

Usage (local, no GPU, no model download -- proves the UI/schema rendering works):
    python scripts/demo.py --no-weights

Usage (Colab / GPU machine, real generations):
    python scripts/demo.py --config configs/full_medlook.yaml --adapter-dir /path/to/final_adapter

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions. This banner is also rendered permanently inside the demo UI itself.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medlook.demo.gradio_app import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config identifying the base model (required unless --no-weights)")
    parser.add_argument("--adapter-dir", default=None, help="Path to a trained LoRA adapter directory (optional)")
    parser.add_argument("--no-weights", action="store_true", help="Skip loading any model; use a mock generator to verify the UI")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.no_weights:
        model, tokenizer = None, None
    else:
        if not args.config:
            parser.error("--config is required unless --no-weights is set")
        from medlook.train.sft import load_model_with_optional_adapter

        model, tokenizer = load_model_with_optional_adapter(args.config, args.adapter_dir)

    demo = build_demo(model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
