"""Default backend: torch SDPA with the math kernel disallowed on CUDA (review.md M3)."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F

from .base import Layout, StepContext


def _to_bhsd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    return t.transpose(1, 2) if layout == "BSHD" else t


def _from_bhsd(t: torch.Tensor, layout: Layout) -> torch.Tensor:
    return t.transpose(1, 2) if layout == "BSHD" else t


class SdpaBackend:
    name = "sdpa"
    requires_no_mask = False

    def __init__(self, disallow_math_on_cuda: bool = True):
        self.disallow_math_on_cuda = disallow_math_on_cuda

    def available(self, device: Any = None) -> bool:
        return True

    def _kernel_ctx(self, q: torch.Tensor):
        if not (self.disallow_math_on_cuda and q.is_cuda):
            return contextlib.nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])

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
        qh, kh, vh = (_to_bhsd(t, layout) for t in (q, k, v))
        with self._kernel_ctx(qh):
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask, dropout_p=0.0, scale=scale)
        return _from_bhsd(out, layout)
