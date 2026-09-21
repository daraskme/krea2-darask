from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppConfig, PathsConfig
from app.models import ModelRegistry
from app.paths import (
    UnsafePathError,
    resolve_in_roots,
    resolve_output,
    validate_model_name,
    validate_output_id,
)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "..",
        "../x.safetensors",
        "..\\x.safetensors",
        "sub/x.safetensors",
        "sub\\x.safetensors",
        "C:x.safetensors",
        "x.safetensors\0",
        ".hidden.safetensors",
        "~home.safetensors",
        "model.ckpt",
        "model.pth",
        "model.bin",
        "model.pt",
        "model.safetensors.ckpt",
        "a" * 300 + ".safetensors",
    ],
)
def test_validate_model_name_rejects(bad: str):
    with pytest.raises(UnsafePathError):
        validate_model_name(bad)


def test_validate_model_name_accepts():
    assert validate_model_name("krea2_turbo_bf16.safetensors") == "krea2_turbo_bf16.safetensors"
    assert validate_model_name("My LoRA v2 (final).SAFETENSORS")


def test_resolve_in_roots(model_tree: Path, tmp_path: Path):
    p = resolve_in_roots("krea2_turbo_bf16.safetensors", [model_tree / "diffusion_models"])
    assert p.is_file()
    with pytest.raises(FileNotFoundError):
        resolve_in_roots("missing.safetensors", [model_tree / "diffusion_models"])
    with pytest.raises(UnsafePathError):
        resolve_in_roots("bad.ckpt", [model_tree / "diffusion_models"])
    # a root that does not exist is skipped, not an error
    with pytest.raises(FileNotFoundError):
        resolve_in_roots("krea2_turbo_bf16.safetensors", [tmp_path / "nope"])


@pytest.mark.parametrize(
    "bad",
    ["", "x.png", "../2026-01-01/000000_1_abcdef.png", "2026-01-01/../x.png", "2026-01-01/000000_1_abcdef.exe",
     "2026-01-01\\000000_1_abcdef.png", "2026-01-01/000000_1_ABCDEF.png"],
)
def test_validate_output_id_rejects(bad: str):
    with pytest.raises(UnsafePathError):
        validate_output_id(bad)


def test_resolve_output(tmp_path: Path):
    root = tmp_path / "outputs"
    (root / "2026-01-01").mkdir(parents=True)
    f = root / "2026-01-01" / "120000_42_abc123.png"
    f.write_bytes(b"x")
    assert resolve_output("2026-01-01/120000_42_abc123.png", root) == f.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_output("2026-01-01/120000_43_abc123.png", root)


def test_registry_scan_only_safetensors(model_tree: Path, tmp_path: Path):
    cfg = AppConfig(paths=PathsConfig(models_root=model_tree, outputs=tmp_path / "out"))
    reg = ModelRegistry(cfg)
    names = [m.name for m in reg.scan("diffusion_models")]
    assert names == ["krea2_turbo_bf16.safetensors"]
    assert "bad.ckpt" not in names
    loras = [m.name for m in reg.scan("loras")]
    assert loras == ["krea2_turbo_4step_rank_64_lora.safetensors", "style_a.safetensors"]
    assert reg.scan_hf_dirs("text_encoders") == ["Qwen3-VL-4B-Instruct"]
    digest = reg.hash_of("loras", "style_a.safetensors", wait=True)
    assert digest and len(digest) == 10
    resp = reg.all_models()
    assert "Qwen3-VL-4B-Instruct" in resp.hf_text_encoders


def test_registry_extra_roots_dedupe(model_tree: Path, tmp_path: Path):
    extra = tmp_path / "comfy" / "models"
    (extra / "loras").mkdir(parents=True)
    (extra / "loras" / "style_a.safetensors").write_bytes(b"dup")
    (extra / "loras" / "style_b.safetensors").write_bytes(b"new")
    cfg = AppConfig(paths=PathsConfig(models_root=model_tree, extra_roots=[extra], outputs=tmp_path / "out"))
    reg = ModelRegistry(cfg)
    loras = reg.scan("loras")
    assert [m.name for m in loras] == [
        "krea2_turbo_4step_rank_64_lora.safetensors", "style_a.safetensors", "style_b.safetensors",
    ]
    # first root wins for duplicates
    assert reg.resolve("loras", "style_a.safetensors").read_bytes() == b"\2" * 64
    assert reg.resolve("loras", "style_b.safetensors").parent.parent == extra.resolve()
