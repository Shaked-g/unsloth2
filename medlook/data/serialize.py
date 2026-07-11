"""Persists converted Unsloth-format samples (which hold in-memory PIL images) to disk
as JSONL + a sibling images directory, and reloads them.

Rather than embedding raw pixel data in JSON, each image is written once to
`{output_dir}/images/` and referenced by relative path. `load_dataset_jsonl`
materializes those paths back into PIL images at load time (e.g. inside `train.py`).
This keeps the on-disk format small, diffable, and easy to inspect by hand.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List

from PIL import Image


def save_dataset_jsonl(samples: Iterable[dict], output_dir: str, split_name: str) -> str:
    images_dir = os.path.join(output_dir, "images", split_name)
    os.makedirs(images_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample_idx, sample in enumerate(samples):
            serializable = _serialize_sample(sample, images_dir, split_name, sample_idx)
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
    return jsonl_path


def _serialize_sample(sample: dict, images_dir: str, split_name: str, sample_idx: int) -> dict:
    out_messages = []
    image_idx = 0
    for message in sample["messages"]:
        out_content = []
        for item in message["content"]:
            if item["type"] == "image":
                # JPEG keeps the prepared dataset far smaller than PNG (critical on
                # disk-constrained machines writing thousands of medical images).
                filename = f"{split_name}_{sample_idx:06d}_{image_idx:02d}.jpg"
                path = os.path.join(images_dir, filename)
                item["image"].convert("RGB").save(path, format="JPEG", quality=85, optimize=True)
                out_content.append({"type": "image_path", "path": path})
                image_idx += 1
            else:
                out_content.append(item)
        out_messages.append({"role": message["role"], "content": out_content})

    return {"messages": out_messages, "meta": sample.get("meta", {})}


def load_dataset_jsonl(jsonl_path: str) -> List[dict]:
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            samples.append(_materialize_sample(raw))
    return samples


def _materialize_sample(raw: dict) -> dict:
    out_messages = []
    for message in raw["messages"]:
        out_content = []
        for item in message["content"]:
            if item["type"] == "image_path":
                out_content.append({"type": "image", "image": Image.open(item["path"])})
            else:
                out_content.append(item)
        out_messages.append({"role": message["role"], "content": out_content})
    return {"messages": out_messages, "meta": raw.get("meta", {})}
