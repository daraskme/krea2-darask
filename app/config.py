"""config.toml loading (pydantic-validated) and path roots."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QuantMode = Literal["bf16", "int8_convrot", "fp8_torchao", "fp8_comfy_kitchen"]
EngineKind = Literal["native", "fake"]

MODEL_KINDS = ("diffusion_models", "text_encoders", "vae", "loras", "upscale_models")
ModelKind = Literal["diffusion_models", "text_encoders", "vae", "loras", "upscale_models"]

REPO_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    models_root: Path = Field(default=REPO_ROOT / "models")
    extra_roots: list[Path] = Field(default_factory=list)
    outputs: Path = Field(default=REPO_ROOT / "outputs")
    inductor_cache: Path = Field(default=REPO_ROOT / ".inductor_cache")


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    dev_cors: bool = False
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])


class DefaultsConfig(BaseModel):
    transformer: str = "krea2_turbo_bf16.safetensors"
    text_encoder: str = "Qwen/Qwen3-VL-4B-Instruct"
    vae: str = "Qwen/Qwen-Image"
    lora_4step: str = "krea2_turbo_4step_rank_64_lora.safetensors"
    upscaler: str = "4x-UltraSharpV2.safetensors"
    preset: str = "fast_4step"
    resolution: str = "832x1216"


class OptimizationConfig(BaseModel):
    quant: QuantMode = "bf16"
    attention: list[str] = Field(default_factory=lambda: ["sdpa"])
    compact_text_tokens: bool = False
    compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    warmup: bool = True
    bsa_enabled: bool = True
    bsa_tau: float = 1.3
    bsa_start_percent: float = 0.2
    bsa_min_tokens: int = 12288
    nag_enabled: bool = False
    text_cache_size: int = 16
    sdpa_disallow_math: bool = True


class AppConfig(BaseModel):
    engine: EngineKind = "native"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    def model_roots(self) -> list[Path]:
        """All model roots (primary first). Only existing directories are returned, resolved."""
        roots: list[Path] = []
        for root in [self.paths.models_root, *self.paths.extra_roots]:
            try:
                resolved = Path(root).expanduser().resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if resolved.is_dir() and resolved not in roots:
                roots.append(resolved)
        return roots

    def kind_dirs(self, kind: ModelKind) -> list[Path]:
        return [root / kind for root in self.model_roots() if (root / kind).is_dir()]


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load config.toml (falls back to config.local.toml overlay if present)."""
    base = REPO_ROOT / "config.toml" if path is None else Path(path)
    data: dict = {}
    if base.is_file():
        with base.open("rb") as fh:
            data = tomllib.load(fh)
    local = base.with_name("config.local.toml")
    if path is None and local.is_file():
        with local.open("rb") as fh:
            _deep_update(data, tomllib.load(fh))
    return AppConfig.model_validate(data)


def _deep_update(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
