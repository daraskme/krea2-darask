"""Deterministic GPU-free engine: seed (+prompt) -> hashed image. Used by tests, Playwright e2e and ``--fake``."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.schemas import GenerateRequest

from .base import EngineOutput, ProgressCallback

_PALETTES = [
    ((20, 24, 40), (120, 80, 200)),
    ((10, 40, 30), (60, 200, 140)),
    ((40, 10, 20), (230, 90, 80)),
    ((30, 30, 10), (230, 200, 60)),
    ((10, 20, 40), (60, 140, 230)),
]


def _digest(*parts: Any) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.digest()


def render_fake_image(seed: int, width: int, height: int, prompt: str, label: str = "") -> Image.Image:
    """Pure function: same (seed, size, prompt) -> byte-identical image."""
    d = _digest(seed, prompt)
    rng = np.random.default_rng(int.from_bytes(d[:8], "little"))
    top, bottom = _PALETTES[d[8] % len(_PALETTES)]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    grad = (1 - y) * np.array(top, np.float32) + y * np.array(bottom, np.float32)
    noise = rng.normal(0.0, 12.0, size=(height, width, 1)).astype(np.float32)
    arr = np.clip(grad + noise, 0, 255).astype(np.uint8)
    arr = np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    # a few deterministic circles so images are visually distinct in the gallery
    for i in range(6):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(min(width, height) // 12, min(width, height) // 4))
        color = tuple(int(c) for c in rng.integers(40, 255, size=3))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=6 + i)
    text = f"fake seed={seed} {width}x{height}" + (f" {label}" if label else "")
    draw.rectangle((0, 0, min(width, 8 * len(text) + 16), 22), fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255))
    return img


class FakeEngine:
    name = "fake"

    def __init__(self, step_delay_s: float = 0.02):
        self.step_delay_s = step_delay_s
        self.warmed_up = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "gpu": None,
            "quant": "none",
            "attention_backends": {"sdpa": {"available": True, "disabled": None, "requires_no_mask": False}},
            "hires": True,
            "bsa": False,
            "nag": True,
            "compile": False,
            "loaded": True,
            "fake": True,
        }

    def warmup(self, progress: ProgressCallback | None = None) -> None:
        self.warmed_up = True
        if progress:
            progress(1, 1, {"phase": "warmup", "message": "fake engine ready"})

    def count_text_tokens(self, prompt: str) -> int:
        # ~1 token per 4 chars, clamped to the model's 512 budget; deterministic & good enough for the UI counter
        return min(512, max(0, (len(prompt) + 3) // 4))

    def generate(self, req: GenerateRequest, seed: int, progress: ProgressCallback) -> list[EngineOutput]:
        outputs: list[EngineOutput] = []
        width, height = req.sampling_size()
        steps = req.hires.steps if req.hires.enabled else req.steps
        for b in range(req.batch_size):
            t0 = time.perf_counter()
            s = seed + b
            for step in range(1, steps + 1):
                if self.step_delay_s:
                    time.sleep(self.step_delay_s)
                progress(step, steps, {"phase": "hires" if req.hires.enabled else "sample", "batch": b})
            label = "hires" if req.hires.enabled else req.preset
            image = render_fake_image(s, width, height, req.prompt, label)
            outputs.append(
                EngineOutput(
                    image=image,
                    seed=s,
                    width=width,
                    height=height,
                    elapsed_s=time.perf_counter() - t0,
                    steps=steps,
                    attention_backend="sdpa",
                    quant="none",
                    bsa_used=False,
                    nag_used=req.nag.enabled,
                    gpu_name=None,
                    text_tokens=self.count_text_tokens(req.prompt),
                )
            )
        return outputs

    def close(self) -> None:
        return None
