from __future__ import annotations

import torch

from app.attention import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    AttentionRegistry,
    BackendUnavailable,
    SolAttnConfig,
    StepContext,
    sol_attn_reason,
)


def _qkv(tokens: int, heads: int = 48, kv_heads: int = 12, dim: int = 128, dtype=torch.bfloat16):
    q = torch.randn(1, tokens, heads, dim, dtype=dtype)
    k = torch.randn(1, tokens, kv_heads, dim, dtype=dtype).repeat_interleave(heads // kv_heads, dim=2)
    v = torch.randn(1, tokens, kv_heads, dim, dtype=dtype).repeat_interleave(heads // kv_heads, dim=2)
    return q, k, v


def test_registry_names_and_default():
    reg = AttentionRegistry()
    assert set(reg.names()) == set(BACKEND_NAMES)
    assert "flash3" not in reg.names() and "flash4" not in reg.names()
    assert DEFAULT_BACKEND == "sdpa"
    status = reg.status()
    assert status["sdpa"]["available"] is True
    assert status["vc_attention"]["available"] is False
    # no GPU here: optional kernels report unavailable instead of raising
    assert status["sage2"]["available"] is False
    assert status["sol_attn"]["available"] is False


def test_select_falls_back_to_sdpa_and_respects_mask():
    reg = AttentionRegistry()
    assert reg.select(["vc_attention", "sage2", "sdpa"], mask_present=False).name == "sdpa"
    assert reg.select(["sage2"], mask_present=True).name == "sdpa"
    assert reg.select(None, mask_present=True).name == "sdpa"


def test_sdpa_backend_matches_reference_and_layouts():
    reg = AttentionRegistry(sdpa_disallow_math=False)
    q, k, v = _qkv(64, dtype=torch.float32)
    out = reg.run(reg.get("sdpa"), q, k, v, layout="BSHD")
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    torch.testing.assert_close(out, ref.transpose(1, 2))
    out2 = reg.run(reg.get("sdpa"), q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), layout="BHSD")
    torch.testing.assert_close(out2, ref)
    mask = torch.ones(1, 1, 1, 64, dtype=torch.bool)
    mask[..., 60:] = False
    out3 = reg.run(reg.get("sdpa"), q, k, v, attn_mask=mask)
    assert out3.shape == q.shape


def test_failing_backend_is_disabled_once_and_logged(caplog):
    class Boom:
        name = "boom"
        requires_no_mask = False
        calls = 0

        def available(self, device=None):
            return True

        def __call__(self, q, k, v, **kw):
            Boom.calls += 1
            raise RuntimeError("kernel exploded")

    reg = AttentionRegistry(sdpa_disallow_math=False)
    reg.register(Boom())
    q, k, v = _qkv(16, dtype=torch.float32)
    with caplog.at_level("WARNING"):
        out = reg.run(reg.get("boom"), q, k, v)
        out2 = reg.run(reg.get("boom"), q, k, v)
    assert out.shape == q.shape and out2.shape == q.shape
    assert Boom.calls == 1
    assert reg.is_disabled("boom")
    assert "disabled for this process" in caplog.text
    assert reg.select(["boom"], mask_present=False).name == "sdpa"


def test_vc_attention_stub_raises_unavailable():
    reg = AttentionRegistry()
    q, k, v = _qkv(16, dtype=torch.float32)
    import pytest

    with pytest.raises(BackendUnavailable):
        reg.get("vc_attention")(q, k, v)


def test_sol_attn_eligibility_order():
    cfg = SolAttnConfig()
    ok = lambda dev: True  # noqa: E731
    q, k, v = _qkv(12288 + 512)
    ctx = StepContext(step=4, steps=8, text_len=512, image_tokens=12288)
    # every rule in ComfyUI order
    assert "mask" in sol_attn_reason(q, k, v, attn_mask=torch.ones(1), step_ctx=ctx, cfg=cfg, is_available=ok)
    early = StepContext(step=1, steps=8)
    assert "start_percent" in sol_attn_reason(q, k, v, attn_mask=None, step_ctx=early, cfg=cfg, is_available=ok)
    small = _qkv(4096 + 512)
    assert "min_tokens" in sol_attn_reason(*small, attn_mask=None, step_ctx=ctx, cfg=cfg, is_available=ok)
    q64 = torch.randn(1, 12800, 48, 64, dtype=torch.bfloat16)
    assert "head_dim" in sol_attn_reason(q64, q64, q64, attn_mask=None, step_ctx=ctx, cfg=cfg, is_available=ok)
    k_small = k[:, :, :12]
    assert "shapes differ" in sol_attn_reason(q, k_small, v, attn_mask=None, step_ctx=ctx, cfg=cfg, is_available=ok)
    qf = q.float()
    assert "dtype" in sol_attn_reason(qf, qf, qf, attn_mask=None, step_ctx=ctx, cfg=cfg, is_available=ok)
    # all tensor rules pass -> device rule (CPU here) is the final blocker
    assert "CUDA" in sol_attn_reason(q, k, v, attn_mask=None, step_ctx=ctx, cfg=cfg, is_available=ok)
    # exact threshold step: 0.2 * 8 = 1.6 -> step 2 is eligible, step 1 is not
    assert "start_percent" not in (sol_attn_reason(q, k, v, attn_mask=None, step_ctx=StepContext(2, 8), cfg=cfg, is_available=ok) or "")
