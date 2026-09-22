from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "outputs").resolve()

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 8189, "open_browser": True, "max_pending_jobs": 8},
    "paths": {
        "model_root": str(PROJECT_ROOT / "models"),
        "transformer_bf16": str(PROJECT_ROOT / "models/diffusion_models/krea2/krea2_turbo_bf16.safetensors"),
        "transformer_int8": str(PROJECT_ROOT / "models/diffusion_models/krea2/krea2_turbo_int8_convrot.safetensors"),
        "raw_int8": str(PROJECT_ROOT / "models/diffusion_models/krea2/krea2_raw_int8_convrot.safetensors"),
        "text_encoder": str(PROJECT_ROOT / "models/text_encoders/qwen3vl_4b_bf16.safetensors"),
        "vae": str(PROJECT_ROOT / "models/vae/qwen_image_vae.safetensors"),
        "qwen_config": str(PROJECT_ROOT / "models/qwen3-vl-4b-instruct"),
        "lora_root": str(PROJECT_ROOT / "models/loras/krea2"),
        "upscaler": str(PROJECT_ROOT / "models/upscale_models/4x-UltraSharpV2_Lite.safetensors"),
    },
    "engine": {
        "default_model": "krea2-turbo-bf16",
        "attention_backend": "sdpa",
        "embedding_cache_size": 4,
        "vae_tiling_threshold": 2049,
    },
    "models": [
        {
            "id": "krea2-turbo-bf16", "name": "Krea 2 Turbo BF16", "kind": "single_file_bf16",
            "path": str(PROJECT_ROOT / "models/diffusion_models/krea2/krea2_turbo_bf16.safetensors"),
            "family": "turbo", "distilled": True,
        },
        {
            "id": "krea2-turbo-int8-convrot", "name": "Krea 2 Turbo INT8 ConvRot",
            "kind": "unsupported_quantized",
            "path": str(PROJECT_ROOT / "models/diffusion_models/krea2/krea2_turbo_int8_convrot.safetensors"),
            "family": "turbo", "distilled": True,
            "reason": "ComfyUI ConvRot INT8 is not a Diffusers-compatible quantization format.",
        },
        {
            "id": "krea2-raw-int8-convrot", "name": "Krea 2 Raw INT8 ConvRot", "kind": "unsupported_quantized",
            "path": str(PROJECT_ROOT / "models/diffusion_models/krea2/krea2_raw_int8_convrot.safetensors"),
            "family": "raw", "distilled": False,
            "reason": "Only the local raw checkpoint is ConvRot INT8; no independent compatible loader is installed.",
        },
    ],
}


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = path or Path(os.environ.get("KREA2_STUDIO_CONFIG", PROJECT_ROOT / "config.local.toml"))
    if path.is_file():
        with path.open("rb") as handle:
            _merge(cfg, tomllib.load(handle))
    if cfg["server"]["host"] not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The server must bind to a loopback address.")
    cfg["server"]["port"] = int(cfg["server"]["port"])
    if not 1 <= cfg["server"]["port"] <= 65535:
        raise ValueError("server.port must be between 1 and 65535")
    return cfg


class SettingsStore:
    DEFAULTS = {
        "width": 1024,
        "height": 1024,
        "preset": "fast4",
        "attention_backend": "sdpa",
        "hires": {"enabled": False, "scale": 1.5, "method": "lanczos", "refine_steps": 8, "denoise_strength": 0.3},
    }

    def __init__(self, path: Path | None = None):
        self.path = path or PROJECT_ROOT / "settings.json"

    def load(self) -> dict[str, Any]:
        data = copy.deepcopy(self.DEFAULTS)
        if self.path.is_file():
            try:
                _merge(data, json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return validate_settings(data)

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        normalized = validate_settings(data)
        self.path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        return normalized


def validate_settings(data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(SettingsStore.DEFAULTS)
    _merge(result, data)
    for dim in ("width", "height"):
        value = int(result[dim])
        if value < 256 or value > 4096 or value % 16:
            raise ValueError(f"{dim} must be a multiple of 16 between 256 and 4096")
        result[dim] = value
    if result["preset"] not in {"turbo8", "fast4", "raw"}:
        raise ValueError("Unknown preset")
    if result["attention_backend"] not in {"auto", "sdpa", "sage2"}:
        raise ValueError("Unknown attention backend")
    hires = result["hires"]
    hires["scale"] = float(hires["scale"])
    hires["refine_steps"] = int(hires["refine_steps"])
    hires["denoise_strength"] = float(hires["denoise_strength"])
    if not 1.0 <= hires["scale"] <= 2.0 or not 0.05 <= hires["denoise_strength"] <= 0.95:
        raise ValueError("Invalid high-resolution settings")
    return result


def safe_output_path(relative: str) -> Path:
    candidate = (OUTPUT_ROOT / relative).resolve()
    if candidate != OUTPUT_ROOT and OUTPUT_ROOT not in candidate.parents:
        raise ValueError("Output path escapes the output directory")
    return candidate
