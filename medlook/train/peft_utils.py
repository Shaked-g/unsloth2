"""PEFT (LoRA/RSLoRA) configuration for MedLook-4B ablations.

Kept separate from `medlook/train/sft.py` so that `PeftSettings` construction and
validation can be unit-tested and used by `--dry-run` config checks without pulling in
unsloth/torch. Those heavy dependencies are imported lazily, inside `apply_peft`, only
when real training actually starts (see PROJECT_BLUEPRINT.md section on hardware
constraints -- local dev machine never loads the 4B model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union


@dataclass
class PeftSettings:
    # Conservative default per the Master Instructions: start with vision layers
    # frozen, language/attention/MLP trainable, RSLoRA r=16 on all-linear.
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = True
    target_modules: Union[str, List[str]] = "all-linear"
    random_state: int = 3407

    def validate(self) -> List[str]:
        issues: List[str] = []
        if self.r <= 0:
            issues.append("r must be positive")
        if self.lora_alpha <= 0:
            issues.append("lora_alpha must be positive")
        if not (0.0 <= self.lora_dropout < 1.0):
            issues.append("lora_dropout must be in [0, 1)")
        if not (
            self.finetune_vision_layers
            or self.finetune_language_layers
            or self.finetune_attention_modules
            or self.finetune_mlp_modules
        ):
            issues.append("at least one finetune_* flag must be true")
        return issues


def peft_settings_from_config(cfg: dict) -> PeftSettings:
    return PeftSettings(
        finetune_vision_layers=cfg.get("finetune_vision_layers", False),
        finetune_language_layers=cfg.get("finetune_language_layers", True),
        finetune_attention_modules=cfg.get("finetune_attention_modules", True),
        finetune_mlp_modules=cfg.get("finetune_mlp_modules", True),
        r=cfg.get("r", 16),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        use_rslora=cfg.get("use_rslora", True),
        target_modules=cfg.get("target_modules", "all-linear"),
        random_state=cfg.get("random_state", 3407),
    )


def apply_peft(model, settings: PeftSettings):
    """Wraps `model` with a LoRA/RSLoRA PEFT adapter via Unsloth's `FastVisionModel`.

    Requires unsloth + torch to be installed; only called on Colab (or a real GPU
    machine), never during local --dry-run or unit tests.
    """
    from unsloth import FastVisionModel

    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=settings.finetune_vision_layers,
        finetune_language_layers=settings.finetune_language_layers,
        finetune_attention_modules=settings.finetune_attention_modules,
        finetune_mlp_modules=settings.finetune_mlp_modules,
        r=settings.r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        use_rslora=settings.use_rslora,
        target_modules=settings.target_modules,
        random_state=settings.random_state,
    )
