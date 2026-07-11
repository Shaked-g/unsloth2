# MedLook-4B — Project Blueprint (v1.1)

**Status**: Ready-to-build, hardware-constrained (Colab Pro for training; local for pipeline validation and demo)
**Supersedes**: `idea.md` (original brainstorm, kept for historical reference)

**Disclaimer (must appear everywhere this project is discussed or shown)**:
> Research prototype only. Not for clinical use, diagnosis, or treatment decisions.

---

## 1. Core Thesis

Take Google's **MedGemma 1.5 4B** (`google/medgemma-1.5-4b-it`, released Jan 13, 2026 under the
Health AI Developer Foundations license — a real, gated Hugging Face model, not hypothetical) and
fine-tune it with Unsloth so that **every response carries a first-class, calibrated visual
strategy decision**: `RELOOK`, `ANSWER_CONFIDENT`, `FLAG_UNCERTAIN`, or `ESCALATE`, with an
explicit confidence value and reason. The model doesn't just answer — it decides *how much visual
work the question deserves* and *how sure it is*, and that decision is trained and evaluated as a
first-class citizen alongside answer accuracy.

## 2. Why This Is a Real Improvement, Not a Rebrand

Meissa-4B (arXiv:2603.09018) already demonstrated agentic medical VLM SFT on Qwen3-VL-4B, matching
or exceeding frontier agents in 10/16 evaluation settings across 13 benchmarks. MedLook-4B is not
"Meissa on a different base." Its distinct, falsifiable claims are:

1. **Stronger medical-native base.** MedGemma 1.5 4B uses a medical-tuned SigLIP encoder and shows
   large gains on anatomical localization (Chest ImaGenome IoU 3.1 → 38.0) and longitudinal
   multi-image tasks (MS-CXR-T macro accuracy 65.7) relative to MedGemma 1 — strengths a generic
   VL-4B backbone does not have out of the box.
2. **Calibrated strategy routing as a first-class, independently-evaluated output.** The ACTION
   decision is graded against a held-out, hand-checked **gold strategy set** that is never used in
   training — not against the same heuristics that generated the training labels. This avoids the
   circularity that would otherwise make "strategy accuracy" a meaningless number.
3. **Pluggable, not Meissa-locked, data.** The primary data source (`open_vqa`: PathVQA, SLAKE,
   VQA-RAD) is fully open with no credentialing. Meissa's interleaved trajectories are an optional
   enrichment adapter with graceful fallback. A purpose-built `uncertainty` adapter supplies the
   FLAG/ESCALATE signal that Meissa's data alone does not emphasize, using an anti-shortcut design
   (every degraded example is paired with its clean counterpart) so the model learns actual
   answerability rather than superficial image statistics.
4. **Ablation discipline.** Every claim is reported across four systems — Base zero-shot,
   Short-SFT, Process-SFT, Full-MedLook — never a single cherry-picked comparison.

If the trained model does not clear the pre-committed success gate (Section 7), that is reported
as a negative result with a concrete next-experiment plan. Numbers are never invented.

## 3. Hardware Reality (non-negotiable)

| Resource | Training | Inference / Demo |
|---|---|---|
| Local machine (MX150 2GB + 16GB RAM) | **Impossible** — never attempted | GGUF CPU demo only (slow, single-digit tok/s) |
| Colab Pro (L4 / A100 preferred, T4 acceptable for light runs) | **Primary training environment** | Also used for real evaluation |
| Checkpointing | Every 100 steps → Google Drive, mandatory | — |
| Multi-turn agentic RL / ART | **Out of scope** — hostile to Colab session limits | — |

Consequence: pure SFT only, on Colab. The local machine's job is to make sure every non-training
part of the system (data pipeline, schema, metrics math, demo UI) is provably correct *before* any
GPU time is spent.

## 4. The Strategy Schema (the spine)

Every trained assistant turn ends in this exact structure (single source of truth:
`medlook/schema.py`):

```text
[STRATEGY]
ACTION: RELOOK | ANSWER_CONFIDENT | FLAG_UNCERTAIN | ESCALATE
CONF: 0.00-1.00
REASON: short explanation
[/STRATEGY]
[PROCESS]
... optional multi-step / multi-image reasoning ...
[/PROCESS]
[FINAL]
Answer. Confidence: X.XX
[/FINAL]
```

- Easy, closed-form cases → short `ANSWER_CONFIDENT` path.
- Medium visual difficulty → `RELOOK` (real multi-image when derivable, textual re-examination
  fallback otherwise).
- High ambiguity, degraded input, or conflicting evidence → `FLAG_UNCERTAIN` or `ESCALATE`.

## 5. Base Model

- **Primary**: `unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit` (Unsloth's pre-quantized mirror,
  avoids re-quantizing 4B weights every Colab session).
- **Fallback**: `google/medgemma-1.5-4b-it` directly via `HF_TOKEN`, if the Unsloth mirror is
  stale or unavailable.
- Built on Gemma 3 architecture (decoder-only, GQA, ≥128K context). Chat template boundary tokens
  for `train_on_responses_only`: `instruction_part="<start_of_turn>user\n"`,
  `response_part="<start_of_turn>model\n"`.
- **Known real risk** (from Google's own model card): MedGemma is not optimized for multi-turn use
  and is more prompt-sensitive than base Gemma 3. This directly motivates starting with
  `finetune_vision_layers=False` and running a dedicated multi-image packing smoke test before any
  full training run (Section 8).

## 6. Data Foundation

| Adapter | Source | Access | Role |
|---|---|---|---|
| `open_vqa` (primary) | PathVQA, SLAKE, VQA-RAD (HF `datasets`) | Fully open | Core answer + easy/medium strategy signal |
| `meissa` (optional) | `CYX1998/Meissa-SFT`, filtered to `interleaved_thinking_images` (~10.5k samples: PathVQA 7,674 / SLAKE 1,698 / VQA-RAD 1,155) | Open, Apache-2.0 | Real multi-step RELOOK trajectories with tool-call-derived crops |
| `uncertainty` (purpose-built) | Programmatic degradations of `open_vqa` images (blur, region crop-away, conflicting text hint), paired with clean counterparts | Generated locally | FLAG_UNCERTAIN / ESCALATE signal with anti-shortcut pairing |

**Curriculum mix (default):** ~45–50% open/short+process, ~30–35% multi-step RELOOK, ~15–20%
uncertainty. Configurable in YAML per ablation.

**Quality gates:** schema-valid, non-empty final answer, 1–3 images/sample, RGB-normalized,
length-capped, class-balance enforced (`FLAG_UNCERTAIN + ESCALATE` ≥ 12% of final mix).

**Decontamination:** n-gram overlap + image perceptual hash against held-out eval subsets and the
gold strategy set.

**Gold strategy evaluation set:** 150–300 hand-reviewed cases, carved out before curriculum mixing,
never eligible for training, covering all four ACTIONs with FLAG/ESCALATE deliberately
over-represented (since they are rarest and highest-value to measure well).

## 7. Evaluation and the Success Gate

Four systems, always reported together: **Base zero-shot | Short-SFT | Process-SFT |
Full-MedLook.**

1. **Answer quality**: exact match + token F1 on VQA-RAD / SLAKE / PathVQA held-out subsets.
2. **Strategy selection**: ACTION accuracy/F1 against the gold strategy set only (never
   training-time heuristics). FLAG precision/recall reported separately.
3. **Calibration / honesty**:
   - Expected Calibration Error (ECE)
   - Overconfident-error rate (wrong answer + high stated confidence)
   - **AURC** (Area Under the Risk-Coverage curve — lower is better; standard selective-prediction
     metric, computed by sorting predictions by descending confidence and integrating cumulative
     error over coverage), plus the full risk-coverage curve for qualitative inspection.

**Hard success gate (pre-committed):** Full-MedLook must (a) improve answer F1 on at least one
primary set **and** (b) improve AURC **or** reduce overconfident-error rate, both versus Base and
versus Short-SFT. Missing the gate is a documented negative result plus a concrete next-experiment
proposal — never a reworded claim of success.

Every public claim ships with qualitative examples: a correct RELOOK flip, a correct FLAG, and at
least one honest failure case.

## 8. Training Recipe

```python
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    max_seq_length=4096,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # conservative start
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    use_rslora=True,
    target_modules="all-linear",
    random_state=3407,
)

data_collator = UnslothVisionDataCollator(
    model, tokenizer,
    resize="min",
    completion_only_loss=True,
    train_on_responses_only=True,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=converted_dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        num_train_epochs=1.5,
        learning_rate=1.5e-4,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="/content/drive/MyDrive/medlook_runs/exp001",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    ),
)
trainer.train(resume_from_checkpoint=True)  # resume=True on reconnect
```

**Required checkpoint before scaling up:** a multi-image packing smoke test — push ~50 real
multi-image `RELOOK` samples through the collator and confirm shapes/masks/loss are sane — and a
minimal-viable-experiment pass (Short-SFT vs Full-MedLook on 2–4k samples) before committing to the
full 12–18k mix.

**Pinned dependency ranges** (verified against Unsloth `v2026.6.x`): `torch>=2.4.0,<2.11.0`;
`transformers>=4.51.3,<=5.5.0` excluding known-broken point releases; `trl>=0.18.2,<=0.24.0,
!=0.19.0`; `peft>=0.18.0,!=0.11.0`; `bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0`;
`accelerate>=0.34.1`; `datasets>=3.4.1,<4.4.0`. See `requirements.txt` for exact pins.

## 9. Deliverables

- LoRA adapters (+ optional merged 16-bit, + GGUF `q4_k_m` for CPU demo)
- Gradio multi-image demo with schema-parsed panels and a permanent disclaimer banner
- Four-system evaluation report (JSON + rendered table) with risk-coverage curves
- Reproducible scripts + one Colab notebook
- Model card (`model_card_template.md`, filled in with real numbers post-training)

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Strategy labels are heuristic and circular to evaluate | Held-out gold strategy set, never used in training or heuristic generation |
| MedGemma not optimized for multi-turn / prompt-sensitive | Vision layers frozen first; dedicated multi-image packing smoke test before full runs |
| Model learns uncertainty as a texture shortcut | `uncertainty` adapter pairs every degraded sample with its clean counterpart |
| Colab disconnects mid-run | `save_steps=100` to Drive + full resume support |
| Meissa download/license friction | `meissa` adapter fails gracefully; pipeline runs in "Meissa-off" mode on `open_vqa` alone |
| Overfitting to the schema format itself | Mix short/long trajectories; completion-only loss; curriculum staging |
| Inflated or invented claims | Pre-committed success gate; four-system table always reported together; qualitative failure cases required |

## 11. Out of Scope (enforced strictly)

- Multi-turn agentic RL / ART / multi-turn GRPO
- PhysioNet/MIMIC credentialed data (documented as a future adapter only)
- Claims of clinical readiness or Meissa-Ultra parity
- Training the 4B model on the local machine
- Any metric presented as real without having actually been computed
