from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import AppConfig, DefaultsConfig, OptimizationConfig, PathsConfig, ServerConfig
from app.engine.fake_engine import FakeEngine
from app.main import create_app


@pytest.fixture
def model_tree(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for kind in ("diffusion_models", "text_encoders", "vae", "loras", "upscale_models"):
        (root / kind).mkdir(parents=True)
    (root / "diffusion_models" / "krea2_turbo_bf16.safetensors").write_bytes(b"\0" * 64)
    (root / "diffusion_models" / "bad.ckpt").write_bytes(b"\0" * 64)
    (root / "loras" / "krea2_turbo_4step_rank_64_lora.safetensors").write_bytes(b"\1" * 64)
    (root / "loras" / "style_a.safetensors").write_bytes(b"\2" * 64)
    (root / "upscale_models" / "4x-UltraSharpV2.safetensors").write_bytes(b"\3" * 64)
    (root / "text_encoders" / "Qwen3-VL-4B-Instruct").mkdir()
    (root / "text_encoders" / "Qwen3-VL-4B-Instruct" / "config.json").write_text("{}")
    return root


@pytest.fixture
def app_config(tmp_path: Path, model_tree: Path) -> AppConfig:
    return AppConfig(
        engine="fake",
        paths=PathsConfig(models_root=model_tree, outputs=tmp_path / "outputs"),
        server=ServerConfig(dev_cors=True),
        defaults=DefaultsConfig(),
        optimization=OptimizationConfig(warmup=False),
    )


@pytest.fixture
def client(app_config: AppConfig):
    app = create_app(app_config, engine=FakeEngine(step_delay_s=0.0))
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c
