"""VC-Attention (arXiv:2609.15810, Nunchux AI) slot.

No public implementation exists as of 2026-09-21. When one is released, implement ``__call__`` here (the processor
already passes ``kv_heads`` / ``rope`` / ``text_len`` so a fused kernel can consume them) and make ``available()``
return True on import success. Watch: https://github.com/nunchux , https://github.com/mit-han-lab
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import BackendUnavailable, Layout, StepContext

CANDIDATE_MODULES = ("vc_attention", "nunchux_attention", "vcattn")


class VcAttentionBackend:
    name = "vc_attention"
    requires_no_mask = True

    def available(self, device: Any = None) -> bool:
        return False

    def probe(self) -> str:
        for mod in CANDIDATE_MODULES:
            try:
                importlib.import_module(mod)
            except Exception:  # noqa: BLE001
                continue
            return f"module {mod!r} found, but no adapter is implemented yet"
        return "no public VC-Attention implementation installed"

    def __call__(
        self,
        q: Any,
        k: Any,
        v: Any,
        *,
        attn_mask: Any = None,
        scale: float | None = None,
        rope: Any = None,
        kv_heads: int | None = None,
        text_len: int | None = None,
        step_ctx: StepContext | None = None,
        layout: Layout = "BSHD",
    ) -> Any:
        raise BackendUnavailable("vc_attention: " + self.probe())
