"""``comfy_kitchen.int8_attention`` backend (alternative to SageAttention). BHSD kernel layout; masks supported."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .base import BackendUnavailable, Layout, StepContext


def _ck() -> Any:
    try:
        return importlib.import_module("comfy_kitchen")
    except Exception:  # noqa: BLE001
        return None


class CkInt8AttentionBackend:
    name = "ck_int8_attention"
    requires_no_mask = True  # the kernel accepts masks, but we only promote it on the maskless path

    def available(self, device: Any = None) -> bool:
        ck = _ck()
        if ck is None or not hasattr(ck, "int8_attention") or not torch.cuda.is_available():
            return False
        try:
            return bool(ck.int8_attention_is_available(torch.device(device) if device is not None else None))
        except Exception:  # noqa: BLE001
            return False

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
        ck = _ck()
        if ck is None:
            raise BackendUnavailable("comfy_kitchen not installed")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise BackendUnavailable(f"int8_attention needs bf16/fp16, got {q.dtype}")
        if layout == "BSHD":
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = ck.int8_attention(q, k, v, scale=scale, attn_mask=attn_mask)
        return out.transpose(1, 2) if layout == "BSHD" else out
