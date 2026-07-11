#!/usr/bin/env python
"""CLI: runs the full local smoke test -- proves every pipeline component works with
zero GPU, zero real model weights, and (beyond whatever pip packages are already
installed) zero network access.

Runs, in order:
  1. pytest (the full test suite)
  2. scripts/prepare_data.py --config configs/smoke.yaml
  3. scripts/train.py --config configs/smoke.yaml --dry-run
  4. scripts/eval.py --mock
  5. medlook.demo.gradio_app.build_demo(...) in-process (never launches a server)

Exits non-zero if any step fails, printing a clear PASS/FAIL summary at the end. This
is the single command referenced by README.md's "prove the local pipeline works"
instructions -- it is intentionally just an orchestrator around the already-built CLIs
and library functions, not a reimplementation of any of their logic.

Usage:
    python scripts/smoke_local.py

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions. This script never loads the real 4B model or trains anything.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
PYTHON = sys.executable


def _run_step(name: str, cmd) -> bool:
    print(f"\n{'=' * 70}\n[SMOKE] {name}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    ok = result.returncode == 0
    print(f"[SMOKE] {name}: {'PASS' if ok else 'FAIL'} (exit code {result.returncode})")
    return ok


def _check_demo_build() -> bool:
    print(f"\n{'=' * 70}\n[SMOKE] Gradio demo builds without weights (in-process)\n{'=' * 70}")
    try:
        from medlook.demo.gradio_app import build_demo, mock_generate, render_panels

        demo = build_demo(model=None, tokenizer=None)
        assert demo is not None
        raw = mock_generate([], "Is there a nodule?")
        render_panels(raw)
        print("[SMOKE] Gradio demo build: PASS")
        return True
    except Exception as exc:
        print(f"[SMOKE] Gradio demo build: FAIL ({exc})")
        return False


def main() -> int:
    sys.path.insert(0, REPO_ROOT)

    results = {}
    results["pytest (full test suite)"] = _run_step(
        "pytest (full test suite)", [PYTHON, "-m", "pytest", "tests/", "-q"]
    )
    results["prepare_data.py --config configs/smoke.yaml"] = _run_step(
        "prepare_data.py --config configs/smoke.yaml",
        [PYTHON, "scripts/prepare_data.py", "--config", "configs/smoke.yaml"],
    )
    # Depends on the prepare_data step above having just (re)written outputs/data/smoke.
    results["train.py --config configs/smoke.yaml --dry-run"] = _run_step(
        "train.py --config configs/smoke.yaml --dry-run",
        [PYTHON, "scripts/train.py", "--config", "configs/smoke.yaml", "--dry-run"],
    )
    results["eval.py --mock"] = _run_step("eval.py --mock", [PYTHON, "scripts/eval.py", "--mock"])
    results["Gradio demo build (--no-weights path)"] = _check_demo_build()

    print(f"\n{'=' * 70}\nSMOKE TEST SUMMARY\n{'=' * 70}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    all_passed = all(results.values())
    print("\n" + ("ALL CHECKS PASSED -- ready for Colab." if all_passed else "SOME CHECKS FAILED -- fix before Colab."))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
