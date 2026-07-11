"""Tests that medlook/export/{merge,gguf}.py import cleanly with zero unsloth/torch/peft
installed (all heavy imports must be lazy, inside the functions). Does not exercise the
actual merge/GGUF export logic -- that requires a real GPU and is only ever run on
Colab."""

from __future__ import annotations

import inspect


def test_merge_module_imports_without_torch_or_unsloth():
    from medlook.export import merge

    assert callable(merge.merge_lora_to_fp16)
    sig = inspect.signature(merge.merge_lora_to_fp16)
    assert list(sig.parameters) == ["config_path", "adapter_dir", "output_dir"]


def test_gguf_module_imports_without_torch_or_unsloth():
    from medlook.export import gguf

    assert callable(gguf.export_gguf)
    assert gguf.DEFAULT_QUANTIZATION == "q4_k_m"
    sig = inspect.signature(gguf.export_gguf)
    assert list(sig.parameters) == ["config_path", "adapter_dir", "output_dir", "quantization"]
