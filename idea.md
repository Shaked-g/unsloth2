**MedLook-4B Project Blueprint**  
*Calibrated Multi-Step “Think-with-Images” Specialization of MedGemma 1.5 4B*

**Version**: 1.0 (July 11, 2026)  
**Status**: Ready-to-build, hardware-constrained (Colab Pro + local GGUF demo)  
**Core Thesis**: Transfer high-quality multi-step interleaved visual trajectories onto MedGemma 1.5 4B, training the model not only *how* to re-examine medical images but **when** re-looking is sufficient versus when it must flag uncertainty or escalate. This produces a more reliable, psychologically honest small medical VLM that improves both answer quality and calibrated strategy selection.

---

# 1. Project Summary & Motivation

### One-sentence pitch
We take Google’s MedGemma 1.5 4B (medical SigLIP encoder + native multi-image / multi-slice / multi-timepoint + strong anatomical localization) and distill open Meissa-style interleaved “think-with-images” trajectories into it with Unsloth FastVision SFT, teaching **calibrated visual strategy selection** under uncertainty for offline medical use.

### Why this is worth building (July 2026)
- Generic medical caption/VQA LoRA is saturated.
- Full multi-environment agentic RL (Meissa-Ultra / ART multi-turn) is infeasible on Colab Pro (time limits + VRAM + statefulness).
- Pure abstention/calibration papers are already numerous.
- **Open gap**: Small (≤4B), fully-offline medical VLMs that can *actively decide* visual strategies (re-look / multi-view / stop + flag low confidence) while leveraging the strongest available medical foundation model.

### Novelty claims (defensible)
1. First strong open transfer of Meissa Framework-II interleaved multi-image trajectories onto MedGemma 1.5 4B using Unsloth selective PEFT + multi-image packing.
2. Explicit process supervision of **strategy routing under uncertainty** (re-look vs answer-confident vs flag/escalate) as first-class output.
3. Measured gains on both standard medical VQA **and** risk-aware metrics (risk-coverage, strategy selection F1, overconfident-error reduction) vs strong baselines.
4. Fully reproducible Unsloth notebooks + GGUF that runs on modest hardware (including the user’s laptop via CPU).

### Expected Outcomes
- LoRA adapters + merged 16-bit (optional) + GGUF.
- Gradio multi-image demo.
- Ablation tables + risk-coverage curves.
- Short blog / arXiv note + HF model card (research prototype only).

---

# 2. Technical Foundations (Verified July 2026)

### Base Model
- **Primary**: `unsloth/medgemma-1.5-4b-it` (or `google/medgemma-1.5-4b-it` wrapped by Unsloth)
- Key MedGemma 1.5 4B capabilities (arXiv 2604.05081 + model card):
  - High-dim: multi-slice CT/MRI + multi-patch WSI
  - Longitudinal multi-timepoint CXR
  - Anatomical bounding-box localization: **IoU 3.1 → 38.0** on Chest ImaGenome (very large absolute gain)
  - MS-CXR-T longitudinal: **macro accuracy 65.7** (≈ +4.6 points over MedGemma 1 4B)
  - Medical document / EHR understanding
  - Built on Gemma 3; Unsloth supports FastVisionModel + GGUF (`unsloth/medgemma-1.5-4b-it-GGUF`)

### Key Technique Stack
- Unsloth `FastVisionModel` + selective PEFT (`finetune_vision_layers`, language, attention, MLP independently)
- `UnslothVisionDataCollator` (multi-image, `completion_only_loss=True`, train-on-responses, resize strategies)
- RSLoRA / QLoRA (4-bit)
- Stratified multi-step trajectory SFT (inspired by Meissa three-tier + process format)
- Strategy tokens + light calibration in the output schema

### Data Foundation
- Open Meissa-SFT (`CYX1998/Meissa-SFT`, 25,018 samples, Apache 2.0, ShareGPT)
  - **Primary subset**: Framework II “Interleaved Thinking with Images”
    - PathVQA: 7,674
    - SLAKE: 1,698
    - VQA-RAD: 1,155
    - ≈ **10,527 multi-step visual trajectories** (zoom / region tools + observations + final answer)
- Secondary: lightly regenerated synthetic multi-image / multi-view / ambiguous examples (for strategy diversity)
- Optional later: MIMIC longitudinal (PhysioNet) if credentials available

---

# 3. Hardware Reality & Constraints

| Resource          | Training                          | Inference / Demo                  |
|-------------------|-----------------------------------|-----------------------------------|
| User laptop (MX150 2GB + 16GB RAM) | **Impossible**                   | GGUF CPU via llama.cpp / Unsloth (slow, single-digit tok/s) |
| Colab Pro         | **Primary** (prefer L4 / A100 40GB; T4 acceptable for lighter runs) | Temporary eval                   |
| Checkpointing     | Mandatory every 100–200 steps → Google Drive | -                                |
| Long multi-turn RL/ART | **Avoid** (hostile to Colab sessions) | -                                |

**Consequence**: Pure SFT (or very short single-turn preference) only. Multi-image packing restricted to practical sequence lengths (target ≤ 4–8 images per sample initially).

---

# 4. Core Design: Strategy Schema (the spine)

Every trajectory teaches the model a **calibrated visual strategy**:

```text
[STRATEGY]
ACTION: RELOOK | ANSWER_CONFIDENT | FLAG_UNCERTAIN | ESCALATE
CONF: 0.0–1.0
REASON: short explanation
[/STRATEGY]

[PROCESS]
... optional intermediate re-look / multi-image reasoning ...
[/PROCESS]

[FINAL]
Answer here. Confidence: X.XX
[/FINAL]
```

- Easy cases → short ANSWER_CONFIDENT path
- Medium visual difficulty → RELOOK (multi-image / simulated zoom)
- High ambiguity or OOD → FLAG_UNCERTAIN or ESCALATE

This is more novel than pure transfer-SFT and more useful than pure abstention: the model actively chooses the right depth of visual processing.

---

# 5. Data Pipeline (detailed)

### 5.1 Load & Filter
```python
from datasets import load_dataset
ds = load_dataset("CYX1998/Meissa-SFT", split="train")
iti = ds.filter(lambda x: x["meta"]["framework"] == "interleaved_thinking_images")
# Expected ~10.5k samples
```

### 5.2 Conversion to Unsloth multi-image chat format
Meissa uses ShareGPT + function_call / observation roles. Convert carefully to Unsloth / Gemma-style multi-turn messages with multiple images.

Recommended structure (list comprehension preferred over `.map` for multi-image stability — Unsloth docs):

```python
def convert_meissa_to_unsloth(sample):
    # 1. Load original images (handle list of paths)
    # 2. Rewrite conversations:
    #    - human → user with image(s) + question
    #    - function_call + observation pairs → intermediate assistant/user turns with new image crops if present
    #    - final gpt → assistant turn that includes the [STRATEGY] ... [FINAL] schema
    # 3. For samples without real crops, fabricate simple multi-view or keep single-image
    # 4. Add curriculum tag (easy/medium/hard based on number of steps or original Meissa difficulty)
    return {"messages": [...], "images": [PIL.Image, ...]}  # Unsloth expects this style
```

**Critical conversion notes**
- MedGemma multi-image is supported but less battle-tested than Qwen3-VL. Start with 1–2 images per sample, expand later.
- Crop observations: either embed the crop as a second image + text description, or convert pure tool calls to text reasoning (“I examined the upper-left region…”).
- Quality filter (keep aggressive):
  - Valid structure
  - Final answer present and non-empty
  - No catastrophic length outliers
  - Prefer samples where multi-step actually changes the answer
- Decontamination: n-gram overlap + image hash against test splits of VQA-RAD / PathVQA / SLAKE.

### 5.3 Synthetic Augmentation (for strategy spine)
Generate ~2–5k additional examples:
- Ambiguous cases (two plausible diagnoses)
- Multi-image longitudinal pairs (public CXR if available)
- Forced high-uncertainty cases (blurred, rare findings, conflicting text hints)
- Teacher (stronger VLM or Gemini-class if budget allows) produces the structured STRATEGY + PROCESS format.

Curriculum: 60% pure Meissa interleaved → 30% multi-image enriched → 10% heavy uncertainty/flag examples.

### 5.4 Target Dataset Size
12k–20k high-quality samples (perfectly manageable for Colab SFT).

---

# 6. Training Recipe (Unsloth-native)

### Stage 0 – Warmup
Standard Unsloth interpretation notebooks (ROCO-style or VQA-RAD single-turn) to confirm FastVision + MedGemma loads correctly.

### Stage 1 – Main SFT (core)
```python
from unsloth import FastVisionModel
import torch
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/medgemma-1.5-4b-it",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
    max_seq_length = 4096,  # adjust after testing multi-image packing
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # start conservative; enable later if needed
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,                 # try 8/16/32
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = True,      # recommended
    target_modules = "all-linear",
    modules_to_save = ["lm_head", "embed_tokens"],  # optional
)

# Converted dataset must be a list or Dataset of {"messages": [...]}  
# with images properly handled as Unsloth expects

data_collator = UnslothVisionDataCollator(
    model, tokenizer,
    resize = "min",                 # or tuple / "max"
    completion_only_loss = True,
    train_on_responses_only = True, # match Gemma chat template boundaries
    # set instruction_part / response_part to MedGemma/Gemma chat tokens
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator,
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,           # L4/A100 friendly
        gradient_accumulation_steps = 8–16,
        warmup_ratio = 0.03,
        num_train_epochs = 1–2,                   # or max_steps
        learning_rate = 1e-4 – 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "/content/drive/MyDrive/medlook_checkpoints",
        report_to = "none",                       # or wandb
        save_strategy = "steps",
        save_steps = 100,                         # CRITICAL for Colab
        save_total_limit = 3,
        # bf16 / fp16 as appropriate for the GPU
    ),
)

trainer.train()
```

**Curriculum training**  
1. Short-answer / easy strategy samples  
2. Full multi-step interleaved  
3. High-uncertainty / FLAG samples

**Ablations to run**
- Base MedGemma 1.5 zero-shot
- Short-answer only SFT (no process/strategy)
- Meissa-format transfer without strategy tags
- Full MedLook with strategy + multi-image

### Stage 2 (optional stretch)
Light preference / DPO or short single-turn GRPO on good vs bad strategy trajectories (correct re-look vs incorrect, good confidence vs overconfident). Keep total run under 1–2 hours continuous so disconnect risk is low.

### Export
```python
model.save_pretrained_merged("medlook-merged", tokenizer, save_method="merged_16bit")
# or
model.save_pretrained_gguf("medlook-gguf", tokenizer, quantization_method="q4_k_m")
# + push adapters to HF
```

---

# 7. Evaluation Suite (makes “we improved” undeniable)

### Core Test Sets
- VQA-RAD, PathVQA, SLAKE (standard + hard subsets)
- OmniMedVQA / MedXpertQA style subsets if available
- Optional longitudinal mini-bench (public pairs or MS-CXR-T style)
- Strategy evaluation set (hand-curated or LLM-generated cases with gold strategy labels: re-look needed vs answerable immediately vs ambiguous)

### Metrics
1. Answer quality: tokenized F1 / exact-match / clinical concept F1 / RadGraph-style if reports
2. Process quality: strategy selection accuracy / F1 (predicted ACTION vs oracle)
3. Calibration / honesty:
   - Risk-coverage curves
   - Accuracy vs confidence
   - Overconfident error rate (wrong answer with high conf)
   - Expected Calibration Error (ECE) or risk-aware variants
4. Efficiency: tokens generated, wall-clock, peak VRAM
5. Multi-image specific: consistency when priors or multi-views are provided

**Required reporting table**  
Base zero-shot | Short-answer SFT | Multi-step SFT | Full MedLook (strategy + multi-image)

Always include qualitative error analysis: cases where re-looking actually flips the answer correctly, vs cases of premature FLAG or hallucinated zooms.

---

# 8. Timeline (realistic for solo + Colab Pro)

| Week | Goal |
|------|------|
| 1    | Environment, Unsloth MedGemma load test, Meissa data download + exploration, convert first 500 samples, baseline zero-shot metrics |
| 2    | Full conversion pipeline + quality filters + first multi-image handling, short SFT run on 2k samples |
| 3–4  | Full 12–18k dataset + main SFT runs + Drive checkpoints + first ablations |
| 5    | Add synthetic strategy/uncertainty examples, second training pass, full evaluation |
| 6    | Polish output format, Gradio demo (multi-image upload → strategy + process + answer), GGUF packaging |
| 7    | Write model card, notebooks, blog/arXiv draft, human preference spot-check (optional) |
| 8    | Buffer for re-runs, better ablations, packaging clean-up |

---

# 9. Deliverables Checklist

**Code / Artifacts**
- `01_data_conversion.ipynb` (or .py) – Meissa → Unsloth multi-image + STRATEGY format
- `02_train_medlook.ipynb` – complete Unsloth training
- `03_eval.ipynb` – metrics + risk-coverage
- `04_demo_gradio.ipynb` / app
- LoRA adapters (HF)
- Optional GGUF
- Config YAMLs + requirements

**Documentation**
- This Blueprint → `PROJECT_BLUEPRINT.md`
- Model card with medical disclaimer
- Training logs / WandB if used

**Disclaimer (must appear everywhere)**  
“Research prototype only. Not for clinical use, diagnosis, or treatment decisions.”

---

# 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Multi-image packing instability on MedGemma | Start single-image dominant, gradually increase; aggressive resize; test collator thoroughly |
| Colab disconnects | `save_steps=100` + Drive mount + resume that loads last checkpoint |
| Thin novelty accusation (“just Meissa on different base”) | Make STRATEGY schema + risk-coverage the headline contribution; rigorous ablations |
| Conversion bugs (zooms/crops) | Fall back to textual “I re-examined region X” + keep original image(s) |
| Overfitting to process format | Mix short + long trajectories; completion-only loss; curriculum |
| Medical safety | Strong disclaimer + prefer high-precision / FLAG behavior |

---

# 11. Full Ready-to-Adapt Notebook Skeleton

Below is a condensed but complete Colab-ready notebook structure. Copy into Colab, enable GPU (L4 preferred), mount Drive, and expand.

```python
# ============================================
# MedLook-4B Training Notebook (Unsloth)
# Colab Pro recommended (L4 / A100)
# ============================================

# 0. Setup
!pip install -U unsloth transformers datasets trl peft accelerate bitsandbytes pillow
# + any other deps

from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = "/content/drive/MyDrive/medlook_runs/exp001"
import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from PIL import Image
import json

# 1. Load model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/medgemma-1.5-4b-it",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    max_seq_length=4096,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,   # start here
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    use_rslora=True,
    target_modules="all-linear",
    random_state=3407,
)

# 2. Data loading + conversion (implement convert carefully)
raw = load_dataset("CYX1998/Meissa-SFT", split="train")
iti = raw.filter(lambda x: x["meta"]["framework"] == "interleaved_thinking_images")

def convert_to_medlook(sample):
    # TODO: robust implementation
    # - parse conversations
    # - load images from sample["images"]
    # - rewrite into messages with proper multi-turn structure
    # - inject [STRATEGY] ... [FINAL] schema on final assistant turn
    # - return {"messages": list_of_dicts}
    pass

# For multi-image stability prefer list comprehension
converted = [convert_to_medlook(s) for s in iti]  # or filtered subset first

# 3. Collator + Trainer
data_collator = UnslothVisionDataCollator(
    model,
    tokenizer,
    resize="min",
    completion_only_loss=True,
    train_on_responses_only=True,
    # set instruction_part / response_part to match MedGemma chat template
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=converted,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=12,
        warmup_ratio=0.03,
        num_train_epochs=1.5,
        learning_rate=1.5e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=CHECKPOINT_DIR,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        report_to="none",
        # bf16=True if supported
    ),
)

# Resume if needed
# trainer.train(resume_from_checkpoint=True)

trainer.train()

# 4. Save
model.save_pretrained(f"{CHECKPOINT_DIR}/lora_adapters")
tokenizer.save_pretrained(f"{CHECKPOINT_DIR}/lora_adapters")
# later: merge + GGUF

print("MedLook training complete. Check Drive for checkpoints.")
```

(You expand `convert_to_medlook` into a production-grade function using the earlier guidelines. Other AI agents can finish the conversion logic.)

---

# 12. AGENT_HANDOFF.md (for another AI / developer)

```markdown
# MedLook-4B Agent Handoff

## Mission
Build MedLook-4B: Unsloth fine-tune of MedGemma 1.5 4B on Meissa interleaved visual trajectories with calibrated strategy selection (`RELOOK / ANSWER_CONFIDENT / FLAG_UNCERTAIN / ESCALATE`).

## Critical Constraints
- Training only on Colab Pro (prefer L4/A100). Never on MX150.
- SFT-only (no multi-turn agentic RL).
- Checkpoint every 100 steps to Drive.
- Keep multi-image count modest initially (1–3 images).
- Research prototype only + medical disclaimer.

## Must-read
- This PROJECT_BLUEPRINT.md (full)
- MedGemma 1.5 model card + arXiv 2604.05081
- Meissa paper (arXiv 2603.09018) + CYX1998/Meissa-SFT dataset card
- Unsloth Vision Fine-tuning docs

## Ordered tasks for the agent
1. Reproduce Unsloth FastVision MedGemma load + simple ROCO/VQA-RAD SFT as smoke test.
2. Write and unit-test robust `convert_meissa_to_medlook` (output Unsloth multi-image chat format + STRATEGY schema).
3. Quality-filter + create train/val split.
4. Run main SFT with ablations (short-answer vs full process).
5. Implement evaluation: answer metrics + strategy selection F1 + risk-coverage.
6. Package adapters + GGUF + Gradio multi-image demo.
7. Write clean model card and short technical note.

## Success criteria
- Clear lift over base MedGemma and short-answer SFT on both accuracy and calibration/strategy metrics.
- Working offline GGUF + local demo that demonstrates multi-step reasoning.
- Reproducible notebooks.

## Forbidden
- Claiming clinical readiness.
- Multi-environment Meissa full clone.
- Heavy multi-turn RL that cannot survive Colab disconnects.
- Exaggerated localization or longitudinal numbers (use exact tables: IoU 3.1→38.0, MS-CXR-T 65.7).
```

---

# 13. Final Recommendation & Next Action

This version of MedLook is:
- Fully compatible with the hardware (Colab Pro + GGUF laptop demo)
- Faithful to the original “Idea 2 / multi-step offline medical VLM” soul
- Stronger in novelty thanks to **calibrated strategy selection** as the spine
- Using accurate, verified July 2026 numbers and open assets
- Scoped to finish with high probability while still producing a functional, interesting, measurable improvement over the latest small medical VLM

**Immediate next step for you**:  
Create a new Colab notebook, paste the skeleton, mount Drive, and start with Task 1–2 (load model + convert the first 200 Meissa interleaved samples into the STRATEGY format). Once that conversion function is solid, training becomes mechanical.

I can also expand any single piece further (complete production-grade data converter, exact Gradio code, metric code for risk-coverage, or a paper-style abstract). Just say which part you want next.

This is a strong, professional, cutting-edge project that respects all constraints while staying true to the original intelligent vision of a useful small multi-step medical vision model. Let’s build MedLook.