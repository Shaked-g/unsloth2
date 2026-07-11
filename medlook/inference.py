"""Shared real-model inference helper, used by both the Gradio demo
(`medlook/demo/gradio_app.py`) and the Colab prediction-generation CLI
(`scripts/generate_predictions.py`) -- there is exactly one place that builds the
multi-image chat message and calls `model.generate`, so inference-time formatting is
guaranteed to match training-time formatting (see `medlook/data/convert.py`).

Requires unsloth + torch; only ever run on Colab / a real GPU machine, never locally.
"""

from __future__ import annotations

from typing import List

from PIL import Image


def generate_with_model(model, tokenizer, images: List[Image.Image], question: str, max_new_tokens: int = 512) -> str:
    """Runs one real multi-image generation and returns the raw decoded text (still in
    MedLook schema format, to be parsed by `medlook.schema.parse`)."""
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)

    content = [{"type": "text", "text": question}] + [{"type": "image", "image": img} for img in images]
    messages = [{"role": "user", "content": content}]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
