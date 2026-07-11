"""Exports a merged MedLook-4B checkpoint to GGUF for CPU inference (llama.cpp,
Ollama, LM Studio) -- this is what the local machine (MX150 2GB) can actually run for
a CPU demo, per PROJECT_BLUEPRINT.md's hardware constraints.

IMPORTANT CAVEAT (do not paper over this): GGUF export for vision-language models is
less mature than for text-only models as of mid-2026 -- `model.save_pretrained_gguf`
may only successfully convert the language backbone, or may fail outright depending on
llama.cpp's current support for the MedGemma/Gemma-3 vision projector. If the direct
Unsloth helper fails, this module surfaces the error rather than silently producing a
broken or text-only file mislabeled as a full export; the human should fall back to
the documented manual path (merge to 16-bit via merge.py, then run llama.cpp's
`convert_hf_to_gguf.py` directly and check its own vision-support status at that time).

Requires unsloth + torch; only ever run on Colab / a real GPU machine, never locally.
"""

from __future__ import annotations

import os

DEFAULT_QUANTIZATION = "q4_k_m"


def export_gguf(config_path: str, adapter_dir: str, output_dir: str, quantization: str = DEFAULT_QUANTIZATION) -> str:
    """Merges the adapter into the base model (in memory) and exports it to GGUF at
    `output_dir` using `quantization` (default q4_k_m -- good size/quality tradeoff for
    CPU inference). Returns `output_dir`. Raises whatever unsloth/llama.cpp raises on
    failure -- see the module-level caveat above; never invents a successful export.
    """
    from medlook.train.sft import load_model_with_optional_adapter

    model, tokenizer = load_model_with_optional_adapter(config_path, adapter_dir)

    os.makedirs(output_dir, exist_ok=True)
    if not hasattr(model, "save_pretrained_gguf"):
        raise RuntimeError(
            "This model class has no save_pretrained_gguf method (not an Unsloth-wrapped "
            "model). Fall back to: merge_lora_to_fp16(...) then llama.cpp's "
            "convert_hf_to_gguf.py directly."
        )
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
    return output_dir
