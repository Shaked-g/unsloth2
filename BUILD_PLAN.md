# MedLook-4B — Build Plan (Executable Checklist)

This is the living, phase-by-phase checklist for building MedLook-4B. See `PROJECT_BLUEPRINT.md`
for the scientific/technical rationale behind each item. Check items off as they are completed;
this file should always reflect the true state of the repo.

Disclaimer: Research prototype only. Not for clinical use, diagnosis, or treatment decisions.

## Phase 0 — Plans (this phase)

- [x] `PROJECT_BLUEPRINT.md` — improved scientific blueprint
- [x] `BUILD_PLAN.md` — this file
- [x] `AGENT_HANDOFF.md` — short handoff for a future agent/dev
- [ ] `model_card_template.md` — created in Phase 5, filled with real numbers post-Colab

## Phase 1 — Package Skeleton + Schema + Fixtures + Tests

- [x] `pyproject.toml`, `requirements.txt`, `requirements-colab.txt`
- [x] `medlook/__init__.py` and subpackage `__init__.py`s
- [x] `medlook/schema.py` — format/parse/validate/round-trip for `[STRATEGY]/[PROCESS]/[FINAL]`
- [x] `tests/fixtures/` — tiny PNGs + ShareGPT-like JSON samples + gold-set sample
- [x] `tests/test_schema.py` green (17 tests)
- [x] **Checkpoint:** `pytest tests/test_schema.py` passes with zero GPU/model dependency

## Phase 2 — Data Pipeline

- [x] `medlook/data/adapters/base.py` — common intermediate record + adapter interface
- [x] `medlook/data/adapters/open_vqa.py` — PathVQA / SLAKE / VQA-RAD (primary, fully open)
- [x] `medlook/data/adapters/meissa.py` — optional, graceful fallback if unavailable
- [x] `medlook/data/adapters/uncertainty.py` — anti-shortcut clean/degraded pairs
- [x] `medlook/data/strategy_labeler.py` — documented, unit-tested sufficiency heuristics
- [x] `medlook/data/gold_strategy_set.py` — held-out gold strategy set builder/loader
- [x] `medlook/data/convert.py` — intermediate record → Unsloth `messages` format
- [x] `medlook/data/filter.py` — quality gates + class-balance enforcement
- [x] `medlook/data/decontaminate.py` — n-gram (overlap-fraction based, not single-gram) +
      image-hash overlap check
- [x] `medlook/data/curriculum.py` — weighted mixer + stratified split
- [x] `medlook/data/serialize.py` — JSONL + on-disk image persistence for prepared datasets
- [x] `scripts/prepare_data.py` — CLI, writes processed dataset(s) + `data_stats.json`
- [x] `tests/test_adapters.py`, `tests/test_curriculum.py`, `tests/test_data_pipeline.py` green
      (54 tests total across Phases 1-2)
- [x] **Checkpoint:** `python scripts/prepare_data.py --config configs/smoke.yaml` runs end-to-end
      on fixtures with no network access required, and prints a class-balance summary
      (verified: 27 train / 7 val samples from fixtures, decontamination and class-balance
      warnings both confirmed to fire correctly)

## Phase 3 — Training Module + Configs + Colab Notebook

- [x] `medlook/train/peft_utils.py` — PEFT config per ablation
- [x] `medlook/train/sft.py` — `FastVisionModel` + `UnslothVisionDataCollator` + `SFTTrainer` wiring
      (all unsloth/torch/trl imports are lazy so config validation runs with zero GPU deps)
- [x] `configs/smoke.yaml`, `configs/short_sft.yaml`, `configs/process_sft.yaml`,
      `configs/full_medlook.yaml`
- [x] `scripts/train.py` — CLI entry point, `--dry-run`, `--packing-smoke-test N`, resume support
- [x] `notebooks/colab_train_eval.ipynb` — Drive mount, install, prepare/load data, multi-image
      packing smoke test cell, train Short/Process/Full, generate real predictions, run eval,
      export, in-notebook demo, and an explicit honesty checklist before reporting results
      (28 cells; validated as well-formed JSON with the expected cell sequence)
- [x] `tests/test_train_config.py` green (peft_utils + sft.validate_config/dry_run, no unsloth/torch needed)
- [x] **Checkpoint (local):** `python scripts/train.py --config configs/smoke.yaml --dry-run`
      validates config/model wiring without downloading the 4B model (verified: reports 27
      train / 7 val samples from the smoke fixtures)
- [ ] **Checkpoint (Colab, human-run):** multi-image packing smoke test on ~50 real samples passes
      before any full training run
- [ ] **Checkpoint (Colab, human-run):** minimal-viable-experiment (2-4k samples, Short vs Full)
      completes and checkpoints to Drive

## Phase 4 — Evaluation Suite

- [x] `medlook/eval/answer.py` — exact match + token F1
- [x] `medlook/eval/strategy.py` — ACTION accuracy/F1 vs gold set, FLAG precision/recall, confusion matrix
- [x] `medlook/eval/calibration.py` — ECE, overconfident-error rate, AURC + risk-coverage curve
- [x] `medlook/eval/runner.py` — orchestrates the four-system report (JSON + rendered table);
      `check_success_gate` requires the candidate to beat EVERY present baseline (never a
      cherry-picked "any baseline" pass); `load_predictions_from_dir` for real Colab generations
- [x] `scripts/eval.py` — CLI entry point (`--mock` for local plumbing validation, `--predictions-dir`
      for real generations), loudly labels mock output as synthetic
- [x] `tests/test_metrics.py` green (32 tests, fixed input vectors with known/hand-verified
      F1/ECE/AURC/strategy-F1 values)
- [x] **Checkpoint:** `python scripts/eval.py --mock` produces a valid four-system report from
      fixtures alone, no GPU required (verified end-to-end, including a passing and a failing
      success-gate scenario)
- [x] `medlook/inference.py` + `scripts/generate_predictions.py` — bridges trained adapters to
      the `--predictions-dir` layout `scripts/eval.py` reads (closes the "trained adapter -> real
      four-system report" loop; not explicitly named in Section 5's repo layout but required for
      that loop to actually work end-to-end on Colab, not just in mock mode)

## Phase 5 — Demo + Export

- [x] `medlook/demo/gradio_app.py` — multi-image upload, schema-parsed panels, permanent
      disclaimer banner (top and bottom), `--no-weights` mock generation path
- [x] `medlook/export/merge.py` — merged 16-bit export helper (lazy unsloth/peft imports)
- [x] `medlook/export/gguf.py` — GGUF `q4_k_m` export helper, with an explicit documented
      caveat about current vision-model GGUF support maturity rather than assuming success
- [x] `scripts/demo.py` — CLI entry point (works with or without trained weights)
- [x] `model_card_template.md` — limitations, disclaimer, citation, placeholders for real eval
      numbers only (never pre-filled with invented numbers)
- [x] `tests/test_demo.py`, `tests/test_export.py` green
- [x] **Checkpoint:** `python scripts/demo.py --no-weights` launches Gradio and renders the schema
      (verified: Blocks construct successfully; mock_generate + render_panels produce correct
      per-profile panels; unparseable generations degrade gracefully instead of crashing)

## Phase 6 — End-to-End Local Validation + Handoff

- [x] `scripts/smoke_local.py` — orchestrates pytest + prepare_data + train --dry-run + eval
      --mock + an in-process Gradio build check, with a clear PASS/FAIL summary
- [x] `README.md` — one page, explicit "Local = pipeline validated; Colab = model trained and
      evaluated" framing, disclaimer, exact commands
- [x] Full `pytest` green (117 tests)
- [x] `tests/test_smoke.py` green (tests smoke_local.py's own PASS/FAIL logic and demo-build
      check directly, without recursively re-invoking the full suite)
- [x] **Final checkpoint:** `python scripts/smoke_local.py` -> `ALL CHECKS PASSED -- ready for
      Colab.` See `README.md` and the notebook's own prerequisites section (Colab GPU type,
      HF token + MedGemma license acceptance, expected per-phase time) for exact next steps.

## After the Real Colab Run

- [ ] Fill `model_card_template.md` and the eval report with real numbers
- [ ] Check the success gate (see `PROJECT_BLUEPRINT.md` Section 7)
- [ ] If the gate fails: diagnose via the ACTION confusion matrix, class balance, multi-image
      packing behavior, and LR/curriculum — propose and implement concrete next experiments
- [ ] If the gate passes: finalize model card, push adapters, record qualitative examples
