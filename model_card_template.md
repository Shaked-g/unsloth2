# MedLook-4B Model Card

> **Research prototype only. Not for clinical use, diagnosis, or treatment decisions.**

Fill in every `TODO` below with real, measured values after a real Colab training +
evaluation run. Never fill a `TODO` with an invented, estimated, or "expected" number.
If a run has not been done yet, leave the placeholder in place rather than guessing.

## Model summary

- **Base model:** `unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit` (fallback:
  `google/medgemma-1.5-4b-it`)
- **Fine-tuning method:** LoRA / RSLoRA via [Unsloth](https://github.com/unslothai/unsloth),
  `r=TODO`, `lora_alpha=TODO`, target modules `TODO`
- **Training data mix:** TODO (link to the `data_stats.json` produced by
  `scripts/prepare_data.py` for the run this checkpoint came from)
- **Ablation profile:** `full_medlook` (see `PROJECT_BLUEPRINT.md` for the schema)
- **Training hardware:** TODO (e.g. Colab Pro, 1x A100 40GB, N hours)
- **Checkpoint date:** TODO

## Intended use

MedLook-4B is a research prototype demonstrating calibrated visual-strategy selection
(RELOOK / ANSWER_CONFIDENT / FLAG_UNCERTAIN / ESCALATE) layered on top of a medical
vision-language model. It is intended for:

- Research into selective prediction / abstention for medical VQA
- Benchmarking calibration and strategy-selection behavior against baselines
- Demonstration purposes (Gradio demo, qualitative examples)

It is **not** intended for, and must never be used for:

- Clinical diagnosis, triage, or treatment decisions
- Any use where an incorrect or overconfident answer could cause patient harm
- Deployment without a licensed clinician in the loop

## Evaluation results

All numbers below are from `scripts/eval.py` run against real model generations on the
held-out sets described in `PROJECT_BLUEPRINT.md`. **Never copy a number here that was
produced with `--mock`** -- mock output is synthetic and exists only to validate the
eval code itself.

### Four-system comparison

| System | Answer EM | Answer F1 | ACTION Acc | ACTION Macro-F1 | FLAG Precision | FLAG Recall | ECE | Overconfident-Error | AURC |
|---|---|---|---|---|---|---|---|---|---|
| Base (zero-shot) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Short-SFT | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Process-SFT | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Full-MedLook | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**Primary success gate:** TODO (PASSED / NOT PASSED -- copy the exact verdict and
reasons from the JSON report's `success_gate` field, do not soften a "NOT PASSED"
result)

### Answer-quality breakdown by held-out set

| Set | Base F1 | Short-SFT F1 | Process-SFT F1 | Full-MedLook F1 |
|---|---|---|---|---|
| VQA-RAD | TODO | TODO | TODO | TODO |
| PathVQA | TODO | TODO | TODO | TODO |
| SLAKE | TODO | TODO | TODO | TODO |

### Qualitative examples

TODO: include 3-5 real (raw, unedited) generations per system on the same input,
covering at least one RELOOK case and one FLAG_UNCERTAIN/ESCALATE case, so a reader can
see the actual behavior rather than only aggregate numbers.

## Limitations

- Strategy labels for the majority of training data are heuristic (see
  `medlook/data/strategy_labeler.py`), not clinician-reviewed; only the held-out gold
  strategy set (`tests/fixtures/gold_strategy_sample.json` in miniature, or its real
  counterpart at `data/gold_strategy_set.json`) is hand-labeled.
- Trained primarily on open VQA-RAD / PathVQA / SLAKE plus optional Meissa-style
  interleaved data; does not include PhysioNet/MIMIC or other credentialed datasets.
- GGUF export for this vision-language architecture may be limited to the language
  backbone depending on current llama.cpp support (see `medlook/export/gguf.py`).
- Never evaluated by a licensed clinician; all "ESCALATE"/"FLAG_UNCERTAIN" behavior is
  a research signal, not a validated clinical safety mechanism.

## Disclaimer

Research prototype only. Not for clinical use, diagnosis, or treatment decisions.

## Citation

TODO: add citation once/if this work is written up.
