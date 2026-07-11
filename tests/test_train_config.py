"""Tests for medlook/train/peft_utils.py and the config-validation / dry-run paths of
medlook/train/sft.py. Deliberately does NOT import unsloth/torch/trl -- these tests
must pass on a machine with no GPU and none of those packages installed."""

from __future__ import annotations

import copy
import os

import pytest
import yaml

from medlook.train import sft
from medlook.train.peft_utils import PeftSettings, peft_settings_from_config

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SMOKE_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "smoke.yaml")


# --- peft_utils.py ---


def test_peft_settings_from_config_uses_defaults_for_missing_keys():
    settings = peft_settings_from_config({})
    assert settings == PeftSettings()


def test_peft_settings_from_config_respects_overrides():
    settings = peft_settings_from_config({"r": 32, "lora_alpha": 64, "finetune_vision_layers": True})
    assert settings.r == 32
    assert settings.lora_alpha == 64
    assert settings.finetune_vision_layers is True


def test_peft_settings_validate_rejects_non_positive_r():
    assert "r must be positive" in PeftSettings(r=0).validate()


def test_peft_settings_validate_rejects_all_finetune_flags_off():
    settings = PeftSettings(
        finetune_vision_layers=False,
        finetune_language_layers=False,
        finetune_attention_modules=False,
        finetune_mlp_modules=False,
    )
    assert any("finetune_" in msg for msg in settings.validate())


def test_peft_settings_validate_passes_on_defaults():
    assert PeftSettings().validate() == []


# --- sft.load_config ---


def test_load_config_reads_smoke_yaml():
    cfg = sft.load_config(SMOKE_CONFIG_PATH)
    assert cfg["profile"] == "full_medlook"
    assert cfg["model"]["base_model"]


# --- sft.validate_config ---


@pytest.fixture()
def smoke_cfg():
    return sft.load_config(SMOKE_CONFIG_PATH)


def test_validate_config_accepts_smoke_config_after_data_prep(smoke_cfg):
    # Requires `python scripts/prepare_data.py --config configs/smoke.yaml` to have been
    # run at least once (its output is checked into outputs/data/smoke as part of the
    # Phase 2 checkpoint). Skip gracefully if that hasn't happened in this environment.
    data_dir = smoke_cfg["data"]["output_dir"]
    if not os.path.exists(os.path.join(REPO_ROOT, data_dir, "train.jsonl")):
        pytest.skip("outputs/data/smoke/train.jsonl not present; run scripts/prepare_data.py first")
    issues = sft.validate_config(smoke_cfg, SMOKE_CONFIG_PATH)
    assert issues == []


def test_validate_config_flags_missing_top_level_keys():
    issues = sft.validate_config({"profile": "full_medlook"}, "<test>")
    assert any("model" in i for i in issues)
    assert any("peft" in i for i in issues)


def test_validate_config_flags_invalid_profile(smoke_cfg):
    cfg = copy.deepcopy(smoke_cfg)
    cfg["profile"] = "not_a_real_profile"
    issues = sft.validate_config(cfg, SMOKE_CONFIG_PATH)
    assert any("profile" in i for i in issues)


def test_validate_config_flags_missing_base_model(smoke_cfg):
    cfg = copy.deepcopy(smoke_cfg)
    cfg["model"]["base_model"] = ""
    issues = sft.validate_config(cfg, SMOKE_CONFIG_PATH)
    assert any("base_model" in i for i in issues)


def test_validate_config_flags_invalid_peft_settings(smoke_cfg):
    cfg = copy.deepcopy(smoke_cfg)
    cfg["peft"]["r"] = -1
    issues = sft.validate_config(cfg, SMOKE_CONFIG_PATH)
    assert any("peft.r must be positive" in i for i in issues)


def test_validate_config_flags_missing_collator_parts(smoke_cfg):
    cfg = copy.deepcopy(smoke_cfg)
    del cfg["collator"]["instruction_part"]
    issues = sft.validate_config(cfg, SMOKE_CONFIG_PATH)
    assert any("instruction_part" in i for i in issues)


def test_validate_config_flags_missing_prepared_dataset(smoke_cfg, tmp_path):
    cfg = copy.deepcopy(smoke_cfg)
    cfg["data"]["output_dir"] = str(tmp_path / "does_not_exist")
    issues = sft.validate_config(cfg, SMOKE_CONFIG_PATH)
    assert any("train.jsonl not found" in i for i in issues)
    assert any("val.jsonl not found" in i for i in issues)


# --- sft.run_training(dry_run=True) ---


def test_run_training_dry_run_raises_on_invalid_config(tmp_path):
    bad_config_path = tmp_path / "bad.yaml"
    bad_config_path.write_text(yaml.safe_dump({"profile": "full_medlook"}), encoding="utf-8")
    with pytest.raises(ValueError):
        sft.run_training(str(bad_config_path), dry_run=True)


def test_run_training_dry_run_reports_sample_counts_after_data_prep(smoke_cfg):
    data_dir = smoke_cfg["data"]["output_dir"]
    if not os.path.exists(os.path.join(REPO_ROOT, data_dir, "train.jsonl")):
        pytest.skip("outputs/data/smoke/train.jsonl not present; run scripts/prepare_data.py first")
    result = sft.run_training(SMOKE_CONFIG_PATH, dry_run=True)
    assert result["status"] == "dry_run_ok"
    assert result["train_samples"] > 0
    assert result["profile"] == "full_medlook"
