from __future__ import annotations

import pytest

from app.presets import (
    BSA_MIN_TOKENS,
    RESOLUTION_PRESETS,
    WORKFLOW_PRESETS,
    bsa_token_eligible,
    cfg_to_guidance,
    denoise_steps,
    hires_edge_warning,
    image_tokens,
    presets_payload,
    total_tokens,
)
from app.schemas import GenerateRequest, LoraSpec, RequestValidationError


def test_resolution_presets_unique_multiples_of_16():
    ids = [p.id for p in RESOLUTION_PRESETS]
    assert len(ids) == len(set(ids))
    assert "832x2048" in ids
    for p in RESOLUTION_PRESETS:
        assert p.width % 16 == 0 and p.height % 16 == 0
    # duplicates from the plan table are represented by swap, so no (w,h) and (h,w) pair both present
    pairs = {(p.width, p.height) for p in RESOLUTION_PRESETS}
    for w, h in pairs:
        if w != h:
            assert (h, w) not in pairs, f"{w}x{h} and its swap both listed"


def test_token_function_single_source():
    assert image_tokens(1024, 1024) == 4096
    assert image_tokens(832, 1216) == 52 * 76
    assert total_tokens(832, 1216, 512) == 52 * 76 + 512
    # single T2I never reaches the BSA threshold; hires 2x of 832x2048 does
    for p in RESOLUTION_PRESETS:
        assert not bsa_token_eligible(p.width, p.height, 512)
    assert bsa_token_eligible(1664, 4096, 512)
    assert bsa_token_eligible(1024 * 2, 1024 * 2, 0) is True
    assert total_tokens(2048, 2048, 0) == 16384 >= BSA_MIN_TOKENS


def test_cfg_to_guidance():
    assert cfg_to_guidance(1.0) == 0.0
    assert cfg_to_guidance(3.5) == 2.5
    assert cfg_to_guidance(0.0) == 0.0
    assert GenerateRequest(cfg=1.0).guidance_scale == 0.0
    assert GenerateRequest(cfg=3.5).guidance_scale == 2.5


def test_denoise_steps_matches_comfyui():
    assert denoise_steps(8, 0.35) == int(8 / 0.35) == 22
    assert denoise_steps(8, 1.0) == 8
    with pytest.raises(ValueError):
        denoise_steps(8, 0.0)


def test_hires_edge_warning():
    assert hires_edge_warning(832, 2048) is not None
    assert hires_edge_warning(832, 1216) is None


def test_presets_payload_shape():
    payload = presets_payload()
    assert {p["id"] for p in payload["workflows"]} == {p.id for p in WORKFLOW_PRESETS}
    hires = next(p for p in payload["workflows"] if p["id"] == "hires_2x")
    assert hires["lora_4step"] is False and hires["denoise"] == 0.35 and hires["steps"] == 8
    assert payload["limits"]["bsa_min_tokens"] == BSA_MIN_TOKENS


def test_width_height_rounded_to_16():
    r = GenerateRequest(width=1000, height=1001)
    assert (r.width, r.height) == (992, 992)


def test_validate_lora4step_bsa_mutual_exclusion():
    with pytest.raises(RequestValidationError):
        GenerateRequest(lora_4step=True, bsa=True).validate()
    # auto BSA that would trigger by token count is also rejected with 4-step LoRA
    with pytest.raises(RequestValidationError):
        GenerateRequest(lora_4step=True, bsa=None, width=2048, height=2048).validate()
    # ok combinations
    GenerateRequest(lora_4step=True, bsa=False, width=2048, height=2048).validate()
    GenerateRequest(lora_4step=False, bsa=True, width=2048, height=2048).validate()
    GenerateRequest(lora_4step=True, bsa=None, width=1024, height=1024).validate()


def test_hires_preset_forces_lora_off_and_bsa_auto():
    r = GenerateRequest(preset="hires_2x", prompt="x", width=832, height=1216, lora_4step=True)
    assert r.hires.enabled is True
    assert r.lora_4step is False
    assert r.sampling_size() == (1664, 2432)
    assert r.bsa_effective() is True
    r.validate()


def test_validate_nag_rules():
    with pytest.raises(RequestValidationError):
        GenerateRequest(nag={"enabled": True}, cfg=3.5).validate()
    with pytest.raises(RequestValidationError):
        GenerateRequest(nag={"enabled": True}, batch_size=2).validate()
    GenerateRequest(nag={"enabled": True}, cfg=1.0).validate()


def test_active_loras():
    r = GenerateRequest(loras=[LoraSpec(name="a.safetensors"), LoraSpec(name="b.safetensors", enabled=False),
                               LoraSpec(name="c.safetensors", strength=0.0)])
    assert [lora.name for lora in r.active_loras()] == ["a.safetensors"]
