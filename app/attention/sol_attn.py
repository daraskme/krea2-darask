"""Block Sparse Attention via ``comfy_kitchen.sol_attn`` (Sol-Attn, adaptive tau) - review.md C4.

Eligibility is decided in ONE function (``sol_attn_reason``) in the same order ComfyUI uses:
mask None -> step/steps >= start_percent -> tokens >= min_tokens -> head_dim 128 -> q.shape == k.shape ->
dtype bf16/fp16 -> ``sol_attn_is_available(device)``.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import torch

from .base import HEAD_DIM, BackendUnavailable, Layout, StepContext

log = logging.getLogger(__name__)


@dataclass
class SolAttnConfig:
    tau: float = 1.3
    start_percent: float = 0.2
    end_percent: float = 1.0
    min_tokens: int = 12288
    token_aug: int = 256  # ComfyUI "extra_tokens" (user workflow: 256)
    topk_ratio: float = 0.0  # 0.0 = sol-attn adaptive threshold


def _ck() -> Any:
    try:
        return importlib.import_module("comfy_kitchen")
    except Exception:  # noqa: BLE001
        return None


def sol_attn_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    step_ctx: StepContext | None,
    cfg: SolAttnConfig,
    layout: Layout = "BSHD",
    is_available: Any = None,
) -> str | None:
    """Why the call must stay dense, or None when sol_attn may run. Pure function (unit-tested without a GPU)."""
    if attn_mask is not None:
        return "attention mask present"
    if step_ctx is None:
        return "no step context"
    if step_ctx.progress < cfg.start_percent:
        return f"step {step_ctx.step}/{step_ctx.steps} before start_percent {cfg.start_percent}"
    if step_ctx.progress > cfg.end_percent:
        return f"step {step_ctx.step}/{step_ctx.steps} after end_percent {cfg.end_percent}"
    tokens = q.shape[1] if layout == "BSHD" else q.shape[2]
    if tokens < cfg.min_tokens:
        return f"{tokens} tokens < min_tokens {cfg.min_tokens}"
    if q.shape[-1] != HEAD_DIM:
        return f"head_dim {q.shape[-1]} != {HEAD_DIM}"
    if q.shape != k.shape or q.shape != v.shape:
        return "q/k/v shapes differ (GQA must be expanded before the kernel)"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {q.dtype} not in (bf16, fp16)"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "mixed dtypes"
    if q.device.type != "cuda":
        return "not on CUDA"
    if is_available is None:
        ck = _ck()
        if ck is None:
            return "comfy_kitchen not installed"
        is_available = ck.sol_attn_is_available
    if not is_available(q.device):
        return "no compiled sol_attn kernel for this GPU"
    return None


class SolAttnBackend:
    name = "sol_attn"
    requires_no_mask = True

    def __init__(self, cfg: SolAttnConfig | None = None):
        self.cfg = cfg or SolAttnConfig()

    def available(self, device: Any = None) -> bool:
        ck = _ck()
        if ck is None or not hasattr(ck, "sol_attn") or not torch.cuda.is_available():
            return False
        try:
            return bool(ck.sol_attn_is_available(torch.device(device) if device is not None else None))
        except Exception:  # noqa: BLE001
            return False

    def reason(self, q, k, v, *, attn_mask, step_ctx, layout: Layout = "BSHD") -> str | None:
        return sol_attn_reason(q, k, v, attn_mask=attn_mask, step_ctx=step_ctx, cfg=self.cfg, layout=layout)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        scale: float | None = None,
        rope: Any = None,
        kv_heads: int | None = None,
        text_len: int | None = None,
        step_ctx: StepContext | None = None,
        layout: Layout = "BSHD",
    ) -> torch.Tensor:
        reason = self.reason(q, k, v, attn_mask=attn_mask, step_ctx=step_ctx, layout=layout)
        if reason is not None:
            raise BackendUnavailable(reason)
        ck = _ck()
        if layout == "BHSD":
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = ck.sol_attn(
            q,
            k,
            v,
            tau=self.cfg.tau,
            scale=scale,
            sink_blocks=[0, 0],
            sink_q=[0, 0],
            topk_ratio=self.cfg.topk_ratio,
            token_aug=self.cfg.token_aug,
        )
        return out.transpose(1, 2) if layout == "BHSD" else out
