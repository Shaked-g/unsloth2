# MedLook-4B — Agent Handoff

Short orientation for any future AI agent or developer picking this project up.

## Mission

Fine-tune `google/medgemma-1.5-4b-it` with Unsloth so every response carries a first-class,
calibrated visual strategy decision (`RELOOK | ANSWER_CONFIDENT | FLAG_UNCERTAIN | ESCALATE` +
confidence), and prove the lift with honest, ablation-backed evaluation. Research prototype only —
never claim clinical readiness.

## Must-read, in order

1. `PROJECT_BLUEPRINT.md` — scientific rationale, novelty claims, hardware constraints, success gate
2. `BUILD_PLAN.md` — the living, phase-by-phase checklist; reflects true current repo state
3. `idea.md` — original brainstorm (historical reference only, superseded by the blueprint)

## Critical constraints (do not violate)

- Training only happens on Colab Pro (L4/A100 preferred). Never attempt real 4B training on the
  local machine.
- SFT only — no multi-turn agentic RL/ART.
- Checkpoint every 100 steps to Google Drive; always support `resume_from_checkpoint`.
- Strategy ACTION evaluation happens **only** against the held-out gold strategy set, never
  against the same heuristics used to generate training labels.
- Every uncertainty-training example has a clean counterpart from the same image/question
  (anti-shortcut pairing) — never ship degraded-only uncertainty data.
- Multi-image samples capped at 1–3 images.
- Every claim is reported across all four ablation systems together (Base | Short-SFT |
  Process-SFT | Full-MedLook). Never report one system in isolation as "the result."
- If the pre-committed success gate (see blueprint Section 7) is not met, report a negative result
  and propose concrete next experiments. Never invent or reword numbers to claim success.
- The medical disclaimer appears in the README, the demo UI, the model card, and the notebook
  header, always.

## Current state

Check the checkboxes in `BUILD_PLAN.md` for exact progress. As of this handoff, **all local
build phases (0-6) are complete**: package skeleton, schema, data pipeline (adapters, filter,
decontamination, curriculum, gold strategy set), training module + configs, evaluation suite,
Gradio demo + export helpers, and `scripts/smoke_local.py` all pass (`python
scripts/smoke_local.py` is green, 112+ pytest tests pass). `notebooks/colab_train_eval.ipynb` is
written and ready to run.

**What has NOT happened yet:** no real Colab training or evaluation run. There are no real
numbers anywhere in this repo -- `model_card_template.md` is still all `TODO`s, and every
metric a human sees before running the notebook is either a unit-test fixture value or explicitly
labeled `--mock`/synthetic output. The next step for whoever picks this up is to actually run
`notebooks/colab_train_eval.ipynb` on Colab and report what comes back, honestly, whether or not
the success gate passes.

## If a library API has moved on

Unsloth, Transformers, TRL, and the MedGemma model repos evolve quickly. If an API in
`PROJECT_BLUEPRINT.md` or `BUILD_PLAN.md` no longer matches what's installed, verify the current
API (via the library's own docs/changelog, not assumption), adapt the code, and note the version
you verified against in `requirements.txt` or an inline comment. Do not silently guess.

## Forbidden

- Claiming clinical usefulness or diagnostic reliability.
- Cloning Meissa's full multi-environment/multi-agent scope.
- Heavy multi-turn RL that cannot survive Colab disconnects.
- Training the 4B model locally.
- Generating or presenting fake/placeholder metrics as real results.
