"""SageAttention 2.2 backend (woct0rdho Windows wheel). Maskless only (review.md C3)."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import torch

from .base import BackendUnavailable, Layout, StepContext

log = logging.getLogger(__name__)

REQUIRED_SYMBOL = "sageattn_qk_int8_pv_fp8_cuda"


class Sage2Backend:
    name = "sage2"
    requires_no_mask = True

    def __init__(self) -> None:
        self._mod: Any = None
        self._checked = False
        self._reason = ""

    def _load(self) -> None:
        if self._checked:
            return
        self._checked = True
        try:
            mod = importlib.import_module("sageattention")
        except Exception as exc:  # noqa: BLE001 - any import failure means "not available"
            self._reason = f"import sageattention failed: {exc}"
            return
        if not hasattr(mod, "sageattn"):
            self._reason = "sageattention has no `sageattn`"
            return
        if not hasattr(mod, REQUIRED_SYMBOL):
            self._reason = f"sageattention is 1.x (missing {REQUIRED_SYMBOL}); SageAttention 2.2 wheel required"
            return
        self._mod = mod

    @property
    def unavailable_reason(self) -> str:
        self._load()
        return self._reason

    def available(self, device: Any = None) -> bool:
        self._load()
        if self._mod is None:
            return False
        if not torch.cuda.is_available():
            return False
        dev = torch.device(device) if device is not None else torch.device("cuda")
        if dev.type != "cuda":
            return False
        major, _minor = torch.cuda.get_device_capability(dev)
        return major >= 8

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
        if attn_mask is not None:
            raise BackendUnavailable("sage2 does not support attention masks (enable compact_text_tokens)")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise BackendUnavailable(f"sage2 needs bf16/fp16, got {q.dtype}")
        self._load()
        if self._mod is None:
            raise BackendUnavailable(self._reason)
        tensor_layout = "NHD" if layout == "BSHD" else "HND"
        return self._mod.sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=False, sm_scale=scale)
