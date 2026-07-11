"""FastVisionModel + UnslothVisionDataCollator + SFTTrainer wiring for MedLook-4B.

Heavy dependencies (unsloth, torch, trl) are imported lazily inside the functions that
actually touch the model, so `--dry-run` (see scripts/train.py) and unit tests can
validate config + prepared-dataset wiring on a machine with no GPU and none of those
packages installed. Real training only ever runs on Colab.

Disclaimer: research prototype only. Not for clinical use, diagnosis, or treatment
decisions.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import yaml

from medlook.data import serialize
from medlook.train.peft_utils import apply_peft, peft_settings_from_config

REQUIRED_TOP_LEVEL_KEYS = ("model", "peft", "train", "collator", "data")
VALID_PROFILES = ("short_sft", "process_sft", "full_medlook")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict, config_path: str = "<config>") -> List[str]:
    """Returns a list of human-readable problems with `cfg`. Empty list means valid.

    Deliberately checks that `data.output_dir/{train,val}.jsonl` exist (i.e. that
    `scripts/prepare_data.py` has already been run) so a Colab run fails in seconds on
    a config typo rather than after a multi-minute model download.
    """
    issues: List[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            issues.append(f"{config_path}: missing required top-level key '{key}'")
    if issues:
        return issues

    profile = cfg.get("profile")
    if profile not in VALID_PROFILES:
        issues.append(f"profile must be one of {VALID_PROFILES}, got {profile!r}")

    model_cfg = cfg["model"]
    if not model_cfg.get("base_model"):
        issues.append("model.base_model is required")

    peft_settings = peft_settings_from_config(cfg["peft"])
    issues.extend(f"peft.{msg}" for msg in peft_settings.validate())

    train_cfg = cfg["train"]
    if not train_cfg.get("output_dir"):
        issues.append("train.output_dir is required (should be Drive-mounted on Colab)")
    if train_cfg.get("save_steps", 100) <= 0:
        issues.append("train.save_steps must be positive")

    collator_cfg = cfg["collator"]
    if not collator_cfg.get("instruction_part") or not collator_cfg.get("response_part"):
        issues.append("collator.instruction_part and collator.response_part are required")

    data_output_dir = cfg["data"].get("output_dir")
    if not data_output_dir:
        issues.append("data.output_dir is required")
    else:
        for split in ("train", "val"):
            split_path = os.path.join(data_output_dir, f"{split}.jsonl")
            if not os.path.exists(split_path):
                issues.append(
                    f"{split}.jsonl not found at {split_path} -- run "
                    f"`python scripts/prepare_data.py --config {config_path}` first"
                )

    return issues


def load_datasets(cfg: dict) -> Tuple[list, list]:
    data_output_dir = cfg["data"]["output_dir"]
    train_path = os.path.join(data_output_dir, "train.jsonl")
    val_path = os.path.join(data_output_dir, "val.jsonl")
    train_dataset = serialize.load_dataset_jsonl(train_path) if os.path.exists(train_path) else []
    val_dataset = serialize.load_dataset_jsonl(val_path) if os.path.exists(val_path) else []
    return train_dataset, val_dataset


def build_model_and_tokenizer(cfg: dict):
    """Loads the base model + tokenizer via Unsloth's `FastVisionModel`.

    Falls back to `model.fallback_base_model` (e.g. the plain HF checkpoint) if the
    pre-quantized Unsloth mirror is unavailable. Requires unsloth + torch; never called
    during --dry-run.
    """
    from unsloth import FastVisionModel

    model_cfg = cfg["model"]
    common_kwargs = dict(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        use_gradient_checkpointing="unsloth",
        max_seq_length=model_cfg.get("max_seq_length", 4096),
    )
    try:
        return FastVisionModel.from_pretrained(model_cfg["base_model"], **common_kwargs)
    except Exception as exc:
        fallback = model_cfg.get("fallback_base_model")
        if not fallback:
            raise
        print(
            f"WARNING: failed to load {model_cfg['base_model']!r} ({exc}); "
            f"falling back to {fallback!r}"
        )
        return FastVisionModel.from_pretrained(fallback, **common_kwargs)


def load_model_with_optional_adapter(config_path: str, adapter_dir: Optional[str] = None):
    """Loads the base model (+ tokenizer) per `config_path` and, if `adapter_dir` is
    given, attaches that trained LoRA adapter. Shared by scripts/demo.py,
    scripts/generate_predictions.py, and the export helpers so there is exactly one
    place that wires up "base model + optional adapter" for inference. Requires
    unsloth + torch (+ peft if adapter_dir is set); never called during --dry-run.
    """
    cfg = load_config(config_path)
    model, tokenizer = build_model_and_tokenizer(cfg)

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)

    return model, tokenizer


def build_data_collator(cfg: dict, model, tokenizer):
    from unsloth.trainer import UnslothVisionDataCollator

    collator_cfg = cfg["collator"]
    return UnslothVisionDataCollator(
        model,
        tokenizer,
        resize=collator_cfg.get("resize", "min"),
        completion_only_loss=collator_cfg.get("completion_only_loss", True),
        train_on_responses_only=collator_cfg.get("train_on_responses_only", True),
        instruction_part=collator_cfg.get("instruction_part"),
        response_part=collator_cfg.get("response_part"),
    )


def build_trainer(cfg: dict, model, tokenizer, train_dataset, eval_dataset=None):
    """Wires FastVisionModel + UnslothVisionDataCollator + SFTTrainer per the
    non-negotiable flags from PROJECT_BLUEPRINT.md (remove_unused_columns=False,
    dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
    completion_only_loss=True). Requires unsloth + trl; never called during --dry-run.
    """
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel

    peft_settings = peft_settings_from_config(cfg["peft"])
    model = apply_peft(model, peft_settings)
    FastVisionModel.for_training(model)

    data_collator = build_data_collator(cfg, model, tokenizer)

    train_cfg = cfg["train"]
    sft_config = SFTConfig(
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        num_train_epochs=train_cfg.get("num_train_epochs", 1.5),
        max_steps=train_cfg.get("max_steps") or -1,
        learning_rate=train_cfg.get("learning_rate", 1.5e-4),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        seed=train_cfg.get("seed", 3407),
        output_dir=train_cfg["output_dir"],
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 4),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset or None,
        args=sft_config,
    )


def run_training(
    config_path: str, dry_run: bool = False, resume_from_checkpoint: Optional[str] = None
) -> dict:
    cfg = load_config(config_path)
    issues = validate_config(cfg, config_path)
    if issues:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {i}" for i in issues))

    train_dataset, val_dataset = load_datasets(cfg)

    if dry_run:
        return {
            "status": "dry_run_ok",
            "config": config_path,
            "profile": cfg.get("profile"),
            "base_model": cfg["model"]["base_model"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "peft": peft_settings_from_config(cfg["peft"]).__dict__,
        }

    model, tokenizer = build_model_and_tokenizer(cfg)
    trainer = build_trainer(cfg, model, tokenizer, train_dataset, val_dataset)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    save_dir = os.path.join(cfg["train"]["output_dir"], "final_adapter")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return {"status": "trained", "output_dir": save_dir}


def run_packing_smoke_test(config_path: str, num_samples: int = 50) -> dict:
    """Loads the real base model, builds the real collator, and runs it directly on
    the first `num_samples` prepared training examples WITHOUT starting a training
    loop. Meant to be run manually on Colab against real (non-fixture) data before
    committing to a full run -- multi-image packing failures are the single most
    likely way to waste hours of Colab compute, and this catches them in under a
    minute. Requires unsloth + torch; never runs locally without a GPU.
    """
    cfg = load_config(config_path)
    issues = validate_config(cfg, config_path)
    if issues:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {i}" for i in issues))

    model, tokenizer = build_model_and_tokenizer(cfg)
    peft_settings = peft_settings_from_config(cfg["peft"])
    model = apply_peft(model, peft_settings)

    from unsloth import FastVisionModel

    FastVisionModel.for_training(model)
    data_collator = build_data_collator(cfg, model, tokenizer)

    train_dataset, _ = load_datasets(cfg)
    subset = train_dataset[:num_samples]
    if not subset:
        raise ValueError("No prepared training samples found; run scripts/prepare_data.py first")
    batch = data_collator(subset)

    image_counts = [
        sum(1 for item in sample["messages"][0]["content"] if item["type"] == "image")
        for sample in subset
    ]
    return {
        "status": "packing_smoke_test_ok",
        "num_samples": len(subset),
        "batch_keys": sorted(batch.keys()),
        "min_images_per_sample": min(image_counts),
        "max_images_per_sample": max(image_counts),
    }
