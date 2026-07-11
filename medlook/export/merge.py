"""Merges a trained LoRA adapter into the base MedGemma weights, producing a
standalone 16-bit model directory that works with plain HuggingFace `transformers`
(no unsloth/peft required at inference time).

Requires unsloth + torch; only ever run on Colab / a real GPU machine, never locally.
"""

from __future__ import annotations

import os


def merge_lora_to_fp16(config_path: str, adapter_dir: str, output_dir: str) -> str:
    """Loads the base model per `config_path`, attaches the LoRA adapter at
    `adapter_dir`, merges it, and saves a standalone 16-bit model to `output_dir`.
    Returns `output_dir`.
    """
    from medlook.train.sft import load_model_with_optional_adapter

    model, tokenizer = load_model_with_optional_adapter(config_path, adapter_dir)

    os.makedirs(output_dir, exist_ok=True)
    # Unsloth's save_pretrained_merged handles the LoRA-merge + shard-and-save; falls
    # back to plain PEFT merge_and_unload + transformers save if unsloth's helper is
    # unavailable on the installed model class (e.g. a non-Unsloth-wrapped checkpoint).
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return output_dir
