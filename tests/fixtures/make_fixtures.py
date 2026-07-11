"""Generates the tiny synthetic fixture images used by the test suite.

Run once with `python tests/fixtures/make_fixtures.py` whenever fixtures need to be
regenerated. Images are deliberately tiny (64x64) and abstract (simple shapes, no real
medical imagery) -- they only need to be distinguishable enough for adapters/tests to
reason about "clean" vs "degraded" variants, not clinically meaningful.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFilter

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(FIXTURES_DIR, "images")


def _save(img: Image.Image, name: str) -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    img.convert("RGB").save(os.path.join(IMAGES_DIR, name))


def make_clean_scan(seed_shape: str, size: int = 64) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(20, 20, 24))
    draw = ImageDraw.Draw(img)
    if seed_shape == "circle":
        draw.ellipse((16, 16, 48, 48), fill=(200, 60, 60))
    elif seed_shape == "square":
        draw.rectangle((14, 14, 50, 50), fill=(60, 160, 200))
    elif seed_shape == "cross":
        draw.rectangle((28, 10, 36, 54), fill=(220, 220, 80))
        draw.rectangle((10, 28, 54, 36), fill=(220, 220, 80))
    # "gold_*" variants use a different palette/position than the training-fixture
    # shapes above so that perceptual-hash decontamination does not spuriously flag
    # overlap between the (disjoint, in the real pipeline) training and held-out gold
    # strategy pools -- see tests/fixtures/gold_strategy_sample.json.
    elif seed_shape == "gold_circle":
        draw.ellipse((8, 24, 40, 56), fill=(80, 200, 120))
    elif seed_shape == "gold_square":
        draw.rectangle((22, 6, 58, 42), fill=(210, 140, 40))
    elif seed_shape == "gold_cross":
        draw.rectangle((6, 26, 58, 34), fill=(150, 90, 220))
        draw.rectangle((28, 4, 36, 60), fill=(150, 90, 220))
    else:
        raise ValueError(f"Unknown seed_shape {seed_shape!r}")
    return img


def make_degraded_scan(seed_shape: str, size: int = 64, blur_radius: float = 6.0) -> Image.Image:
    clean = make_clean_scan(seed_shape, size=size)
    return clean.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def make_crop(seed_shape: str, size: int = 64) -> Image.Image:
    """Simulates a 'zoomed-in region' crop derived from the clean scan."""
    clean = make_clean_scan(seed_shape, size=size)
    box = (size // 4, size // 4, size - size // 4, size - size // 4)
    return clean.crop(box).resize((size, size))


def main() -> None:
    shapes = ["circle", "square", "cross"]
    for shape in shapes:
        _save(make_clean_scan(shape), f"clean_{shape}.png")
        _save(make_degraded_scan(shape), f"degraded_{shape}.png")
        _save(make_crop(shape), f"crop_{shape}.png")

    # Disjoint image pool for the held-out gold strategy set fixture.
    gold_shapes = ["gold_circle", "gold_square", "gold_cross"]
    for shape in gold_shapes:
        _save(make_clean_scan(shape), f"{shape}.png")
        _save(make_degraded_scan(shape), f"{shape}_degraded.png")

    print(f"Wrote fixture images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
