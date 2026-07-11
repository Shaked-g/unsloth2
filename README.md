# MedLook-4B

> **Research prototype only. Not for clinical use, diagnosis, or treatment decisions.**

MedLook-4B is a research prototype that fine-tunes `google/medgemma-1.5-4b-it` (via
[Unsloth](https://github.com/unslothai/unsloth)) so that every response carries a
first-class **calibrated visual-strategy decision** -- `RELOOK`, `ANSWER_CONFIDENT`,
`FLAG_UNCERTAIN`, or `ESCALATE` -- alongside its answer, and asks whether that
decision is measurably better calibrated than a base model or a plain short-answer SFT
baseline. See [`PROJECT_BLUEPRINT.md`](PROJECT_BLUEPRINT.md) for the full scientific
rationale, and [`BUILD_PLAN.md`](BUILD_PLAN.md) for the current build status.

## Local machine vs. Colab: read this first

| | **Local machine (this repo, right now)** | **Colab (Pro, L4/A100 preferred)** |
|---|---|---|
| Does | Pipeline validation only: schema tests, adapters on tiny fixtures, `--dry-run`, `--mock`, `--no-weights` demo | Real data prep at scale, real training, real evaluation, real export |
| Never does | Load or train the real 4B model | — |
| Entry point | `python scripts/smoke_local.py` | `notebooks/colab_train_eval.ipynb` |

**Do not attempt to train or run the real 4B model on a machine without a real GPU**
(e.g. an MX150 2GB laptop GPU). Every script that can touch the real model imports
`unsloth`/`torch` lazily and only inside the code paths that need them, so local
pipeline validation never requires those packages to be installed.

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
python scripts/smoke_local.py
```

`smoke_local.py` runs the full test suite, then `prepare_data.py`, `train.py --dry-run`,
`eval.py --mock`, and builds the Gradio demo -- all against tiny checked-in fixtures,
with zero GPU, zero real model weights, and no training. A clean run means the pipeline
itself is correct and ready for a real Colab run.

## Quick start (Colab)

1. Accept the [MedGemma license](https://huggingface.co/google/medgemma-1.5-4b-it) and
   get a Hugging Face token.
2. Open `notebooks/colab_train_eval.ipynb` on a GPU runtime (L4 or A100 preferred).
3. The notebook clones **https://github.com/Shaked-g/unsloth2**, builds the gold strategy
   set if needed, prepares the ~4k MVP mix to Drive, trains, evaluates, and exports.

Store your HF token as a Colab secret named `HF_TOKEN` (or paste via getpass). Never
commit tokens into the repo.

## Repository layout

```text
medlook/
  schema.py            # [STRATEGY]/[PROCESS]/[FINAL] format -- single source of truth
  inference.py          # shared real-model generation helper (demo + prediction CLI)
  data/                 # adapters (open_vqa, meissa, uncertainty), filter, decontaminate,
                         # curriculum mixing, gold strategy set, convert, serialize
  train/                # PEFT config + FastVisionModel/SFTTrainer wiring (lazy unsloth import)
  eval/                 # answer (EM/F1), strategy (ACTION F1), calibration (ECE/AURC), runner
  export/               # merged 16-bit + GGUF export helpers
  demo/                 # Gradio multi-image demo
configs/                 # smoke / short_sft / process_sft / full_medlook ablations
scripts/                 # prepare_data, train, eval, generate_predictions, demo, smoke_local
notebooks/               # colab_train_eval.ipynb -- the only place real training happens
tests/                   # full local test suite + tiny synthetic fixtures
```

## The four-system comparison

Every claim this project makes is ablation-backed against four systems, never just one
number in isolation:

1. **Base** -- zero-shot MedGemma 1.5 4B
2. **Short-SFT** -- SFT on `[FINAL]` only (plain short-answer baseline)
3. **Process-SFT** -- SFT on `[PROCESS]` + `[FINAL]` (no strategy decision)
4. **Full-MedLook** -- SFT on `[STRATEGY]` + `[PROCESS]` + `[FINAL]` (the actual proposal)

`scripts/eval.py` scores answer quality (EM, token F1), strategy selection (ACTION
accuracy/F1 + FLAG precision/recall against a held-out, hand-labeled gold strategy set
that is never used for training), and calibration (ECE, overconfident-error rate,
AURC/risk-coverage). The **primary success gate** requires Full-MedLook to beat
*every* baseline present on both an answer-quality metric and a calibration metric --
see `medlook/eval/runner.py::check_success_gate`. Run `python scripts/eval.py --mock`
locally to see this reported end-to-end on synthetic data; never quote `--mock` output
as a real result.

## Honesty rules (non-negotiable)

- Never invent or round-up a metric. If a Colab run doesn't pass the success gate,
  report that and diagnose it (see the notebook's wrap-up checklist) -- don't rerun
  with a different seed hoping for a better number.
- `--mock` and fixture-based results exist only to prove the code is correct; they are
  never a substitute for a real evaluation.
- This is a research prototype. It is not, and must never be presented as, a clinical
  tool.

## Status

See [`BUILD_PLAN.md`](BUILD_PLAN.md) for the current phase-by-phase checklist. As of
this writing: the full local pipeline (data adapters through the Gradio demo) is built
and green (`python scripts/smoke_local.py` passes); no real Colab training run has
been performed yet, so no real evaluation numbers exist -- `model_card_template.md` is
intentionally still full of `TODO`s.
