"""Multi-image Gradio demo for MedLook-4B.

Parses every model generation through `medlook.schema.parse` (the single schema
source of truth) and renders it into three panels: Strategy, Process, Final Answer.
Works in two modes:

  --no-weights   Uses a deterministic mock generator so the UI/schema rendering can be
                 verified with zero GPU, zero model download (see scripts/demo.py and
                 the Phase 5 local checkpoint in BUILD_PLAN.md).
  (default)      Loads a real base model + optional LoRA adapter via Unsloth and
                 generates real completions. Requires unsloth + torch + a GPU.

A permanent disclaimer banner is rendered at both the top and bottom of the page and
cannot be dismissed -- this is a hard, non-negotiable requirement, not a style choice.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PIL import Image

from medlook import DISCLAIMER
from medlook.inference import generate_with_model
from medlook.schema import Action, MedLookResponse, SchemaError, parse

MAX_IMAGES = 3

ACTION_BADGE = {
    Action.ANSWER_CONFIDENT.value: "🟢 ANSWER_CONFIDENT",
    Action.RELOOK.value: "🔍 RELOOK",
    Action.FLAG_UNCERTAIN.value: "🟡 FLAG_UNCERTAIN",
    Action.ESCALATE.value: "🔴 ESCALATE",
}


def mock_generate(images: List[Image.Image], question: str) -> str:
    """Deterministic, clearly-labeled placeholder generation used with --no-weights.
    Exists purely to exercise the UI + schema-rendering path without a model."""
    n = len(images)
    if n == 0:
        action, reason = Action.FLAG_UNCERTAIN.value, "No image was provided."
        final = "Cannot answer without an image."
        conf = 0.20
    elif n >= 2:
        action, reason = Action.RELOOK.value, "Multiple images provided; comparing regions across them."
        final = "[MOCK] Based on comparing the provided images, a plausible answer would go here."
        conf = 0.65
    else:
        action, reason = Action.ANSWER_CONFIDENT.value, "Single clear image, directly answerable question."
        final = "[MOCK] This is a placeholder answer demonstrating the schema -- no real model is loaded."
        conf = 0.75
    return (
        "[STRATEGY]\n"
        f"ACTION: {action}\n"
        f"CONF: {conf:.2f}\n"
        f"REASON: {reason}\n"
        "[/STRATEGY]\n"
        "[PROCESS]\n"
        f"(mock) Received {n} image(s) and question: {question!r}\n"
        "[/PROCESS]\n"
        "[FINAL]\n"
        f"{final} Confidence: {conf:.2f}\n"
        "[/FINAL]"
    )


def render_panels(raw_text: str) -> Tuple[str, str, str, str]:
    """Returns (strategy_md, process_md, final_md, raw_text) for display. Never raises
    -- an unparseable generation is shown verbatim in the Final panel with a warning
    rather than crashing the demo."""
    try:
        response: Optional[MedLookResponse] = parse(raw_text, strict=False)
    except SchemaError as exc:
        warning = f"**Could not parse a [FINAL] block from the generation** ({exc}).\n\nRaw output shown below."
        return "_(no strategy block)_", "_(no process block)_", f"{warning}\n\n---\n\n{raw_text}", raw_text

    if response.has_strategy:
        s = response.strategy
        badge = ACTION_BADGE.get(s.action.value if isinstance(s.action, Action) else str(s.action), str(s.action))
        strategy_md = f"### {badge}\n**Confidence:** {s.conf:.2f}\n\n**Reason:** {s.reason}"
    else:
        strategy_md = "_(this ablation profile has no [STRATEGY] block)_"

    process_md = f"```\n{response.process}\n```" if response.has_process else "_(no [PROCESS] block)_"

    final_md = f"### Answer\n{response.final_answer}\n\n**Confidence:** {response.final_confidence:.2f}"

    return strategy_md, process_md, final_md, raw_text


def build_demo(model=None, tokenizer=None, max_new_tokens: int = 512):
    """Builds the Gradio Blocks app. `model`/`tokenizer` are None in --no-weights mode."""
    import gradio as gr

    weights_loaded = model is not None and tokenizer is not None

    def _run(files, question):
        if not question or not question.strip():
            return "_(enter a question)_", "_(enter a question)_", "**Please enter a question.**", ""

        images: List[Image.Image] = []
        for f in (files or [])[:MAX_IMAGES]:
            path = f if isinstance(f, str) else getattr(f, "name", None)
            if path:
                images.append(Image.open(path).convert("RGB"))

        if weights_loaded:
            raw = generate_with_model(model, tokenizer, images, question, max_new_tokens=max_new_tokens)
        else:
            raw = mock_generate(images, question)

        return render_panels(raw)

    banner = f"## ⚠️ {DISCLAIMER}"
    mode_note = (
        "**Mode:** real model generation" if weights_loaded else "**Mode:** NO WEIGHTS LOADED (mock generation only)"
    )

    with gr.Blocks(title="MedLook-4B Demo") as demo:
        gr.Markdown(banner)
        gr.Markdown(mode_note)
        gr.Markdown(f"Upload 1-{MAX_IMAGES} images (multi-image supports the RELOOK strategy).")

        with gr.Row():
            with gr.Column():
                files = gr.File(
                    label=f"Image(s) (up to {MAX_IMAGES})",
                    file_count="multiple",
                    file_types=["image"],
                )
                question = gr.Textbox(label="Question", placeholder="e.g. Is there a nodule in the right lung?")
                run_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                strategy_out = gr.Markdown(label="Strategy")
                process_out = gr.Markdown(label="Process")
                final_out = gr.Markdown(label="Final Answer")
                with gr.Accordion("Raw model output", open=False):
                    raw_out = gr.Textbox(label="Raw generation", interactive=False)

        run_btn.click(_run, inputs=[files, question], outputs=[strategy_out, process_out, final_out, raw_out])

        gr.Markdown(banner)

    return demo
