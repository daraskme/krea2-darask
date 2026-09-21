from __future__ import annotations

import json
from pathlib import Path

import piexif
import piexif.helper
import pytest
from PIL import Image, PngImagePlugin

from app.metadata import (
    JPEG_APP1_LIMIT,
    PNG_TEXT_LIMIT,
    build_parameters,
    parameters_to_request,
    parse_parameters,
    quote,
    read_image,
    write_image,
)
from app.schemas import GenerateRequest, GenerateResult, Krea2GuiMetadata, LoraSpec


def _meta(**overrides) -> Krea2GuiMetadata:
    fields = dict(
        prompt="a cat, sitting: on a \"mat\"\nsecond line",
        negative_prompt="blurry, low quality",
        width=832,
        height=1216,
        steps=4,
        cfg=1.0,
        seed=1234,
        transformer="krea2_turbo_bf16.safetensors",
        loras=[LoraSpec(name="style_a.safetensors", strength=0.7, hash="abcdef0123")],
    )
    req = GenerateRequest(**{**fields, **overrides})
    res = GenerateResult(
        output_id="2026-01-01/120000_1234_abc123.png", seed=1234, width=832, height=1216, elapsed_s=1.5,
        steps=4, engine="fake", attention_backend="sdpa", quant="bf16", bsa_used=False, nag_used=False,
        transformer_hash="0123456789", lora_hashes={"style_a.safetensors": "abcdef0123"},
    )
    return Krea2GuiMetadata(request=req, result=res)


def test_quote_a1111_compat():
    assert quote("plain") == "plain"
    assert quote("has, comma") == '"has, comma"'
    assert quote("a:b") == '"a:b"'


def test_build_parameters_format():
    meta = _meta()
    text = build_parameters(meta.request, meta.result)
    lines = text.split("\n")
    assert lines[0].startswith("a cat")
    assert any(line.startswith("Negative prompt: ") for line in lines)
    last = lines[-1]
    parsed = parse_parameters(text)
    assert len(parsed.settings) >= 3
    assert parsed.settings["Steps"] == "4"
    assert parsed.settings["CFG scale"] == "1.0"  # ComfyUI cfg, not diffusers guidance
    assert parsed.settings["Seed"] == "1234"
    assert parsed.settings["Size"] == "832x1216"
    assert "Lora hashes" in parsed.settings
    assert "style_a: abcdef0123" in parsed.settings["Lora hashes"]
    assert "Version: krea2-darask" in last
    req = parameters_to_request(parsed)
    assert req["seed"] == 1234 and req["width"] == 832 and req["cfg"] == 1.0
    assert req["prompt"].startswith("a cat")


@pytest.mark.parametrize("fmt", ["png", "jpg", "webp"])
def test_round_trip(tmp_path: Path, fmt: str):
    meta = _meta()
    img = Image.new("RGB", (64, 96), (10, 20, 30))
    path = tmp_path / f"img.{fmt}"
    write_image(img, path, meta, fmt=fmt)
    result = read_image(path)
    assert result.source == "krea2gui"
    assert result.krea2gui is not None
    assert result.krea2gui.request.prompt == meta.request.prompt
    assert result.krea2gui.request.loras[0].strength == 0.7
    assert result.krea2gui.result.seed == 1234
    assert result.request["seed"] == 1234
    assert result.parameters and "Steps: 4" in result.parameters
    assert not result.warnings


def test_png_has_only_parameters_and_krea2gui(tmp_path: Path):
    meta = _meta()
    path = tmp_path / "x.png"
    write_image(Image.new("RGB", (32, 32)), path, meta, fmt="png")
    with Image.open(path) as im:
        keys = set(im.text)
    assert "parameters" in keys and "krea2gui" in keys
    assert "prompt" not in keys and "workflow" not in keys  # no synthetic ComfyUI JSON (review.md M5)
    with Image.open(path) as im:
        total = sum(len(v.encode("utf-8")) for v in im.text.values())
    assert total <= PNG_TEXT_LIMIT


def test_jpeg_exif_within_app1_limit_and_unicode_usercomment(tmp_path: Path):
    meta = _meta(prompt="x" * 20_000)
    path = tmp_path / "x.jpg"
    write_image(Image.new("RGB", (32, 32)), path, meta, fmt="jpg")
    exif = piexif.load(str(path))
    raw = exif["Exif"][piexif.ExifIFD.UserComment]
    assert raw.startswith(b"UNICODE\0")
    text = piexif.helper.UserComment.load(raw)
    assert text.startswith("x" * 100)
    assert len(piexif.dump(exif)) <= JPEG_APP1_LIMIT
    result = read_image(path)
    assert result.source in ("krea2gui", "parameters")
    assert result.request["prompt"].startswith("xxxx")


def test_parameters_only_png_fallback(tmp_path: Path):
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        "hello world\nNegative prompt: bad\nSteps: 8, Sampler: Euler, Schedule type: Simple, CFG scale: 3.5, "
        "Seed: 7, Size: 1024x768, Model: krea2_raw_bf16",
    )
    path = tmp_path / "a1111.png"
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    result = read_image(path)
    assert result.source == "parameters"
    assert result.request["prompt"] == "hello world"
    assert result.request["negative_prompt"] == "bad"
    assert result.request["steps"] == 8 and result.request["cfg"] == 3.5 and result.request["seed"] == 7
    assert result.request["width"] == 1024 and result.request["height"] == 768


def test_comfyui_prompt_best_effort(tmp_path: Path):
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea2_turbo_int8_convrot.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "comfy prompt", "clip": ["9", 0]}},
        "3": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"lora_name": "krea2_turbo_4step_rank_64_lora.safetensors", "strength_model": 1.0, "model": ["1", 0]}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 832, "height": 2048, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"seed": 99, "steps": 8, "cfg": 1.0, "sampler_name": "euler",
                                                    "scheduler": "simple", "denoise": 1.0, "positive": ["2", 0]}},
    }
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", json.dumps(prompt))
    path = tmp_path / "comfy.png"
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    result = read_image(path)
    assert result.source == "comfyui"
    assert result.request["prompt"] == "comfy prompt"
    assert result.request["seed"] == 99 and result.request["steps"] == 8
    assert result.request["width"] == 832 and result.request["height"] == 2048
    assert result.request["transformer"] == "krea2_turbo_int8_convrot.safetensors"
    assert any(lora["name"].startswith("krea2_turbo_4step") for lora in result.request["loras"])


def test_invalid_krea2gui_chunk_falls_back_to_parameters(tmp_path: Path):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "p\nSteps: 4, CFG scale: 1.0, Seed: 1, Size: 512x512")
    info.add_text("krea2gui", '{"schema_version": 1, "request": {"width": "oops"}}')
    path = tmp_path / "bad.png"
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    result = read_image(path)
    assert result.source == "parameters"
    assert result.warnings and "invalid" in result.warnings[0]
    assert result.request["seed"] == 1
