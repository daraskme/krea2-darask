"""Attention backend registry + runtime selection.

* names: sdpa, sage2, sol_attn, ck_int8_attention, vc_attention (flash3/4 intentionally absent)
* default = sdpa; maskless-only backends are eligible only when ``attn_mask is None`` (compact_text_tokens=True)
* a backend that raises ValueError / RuntimeError once is disabled for the process and logged
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AttentionBackend, BackendUnavailable, Layout, StepContext
from .ck_int8 import CkInt8AttentionBackend
from .sage2 import Sage2Backend
from .sdpa import SdpaBackend
from .sol_attn import SolAttnBackend, SolAttnConfig
from .vc_attention import VcAttentionBackend

log = logging.getLogger(__name__)

DEFAULT_BACKEND = "sdpa"
BACKEND_NAMES = ("sdpa", "sage2", "sol_attn", "ck_int8_attention", "vc_attention")


class AttentionRegistry:
    def __init__(self, sol_cfg: SolAttnConfig | None = None, sdpa_disallow_math: bool = True):
        self._backends: dict[str, AttentionBackend] = {}
        self._disabled: dict[str, str] = {}
        self.register(SdpaBackend(disallow_math_on_cuda=sdpa_disallow_math))
        self.register(Sage2Backend())
        self.register(SolAttnBackend(sol_cfg))
        self.register(CkInt8AttentionBackend())
        self.register(VcAttentionBackend())

    # -- registration ------------------------------------------------------------------------------
    def register(self, backend: AttentionBackend) -> None:
        self._backends[backend.name] = backend

    def names(self) -> list[str]:
        return list(self._backends)

    def get(self, name: str) -> AttentionBackend:
        try:
            return self._backends[name]
        except KeyError:
            raise KeyError(f"unknown attention backend {name!r}; known: {self.names()}") from None

    # -- status ------------------------------------------------------------------------------------
    def disable(self, name: str, reason: str) -> None:
        if name not in self._disabled:
            log.warning("attention backend %s disabled for this process: %s", name, reason)
        self._disabled[name] = reason

    def is_disabled(self, name: str) -> bool:
        return name in self._disabled

    def status(self, device: Any = None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, backend in self._backends.items():
            try:
                avail = bool(backend.available(device))
            except Exception as exc:  # noqa: BLE001
                avail = False
                self.disable(name, f"available() raised: {exc}")
            out[name] = {
                "available": avail,
                "disabled": self._disabled.get(name),
                "requires_no_mask": backend.requires_no_mask,
            }
        return out

    # -- selection ---------------------------------------------------------------------------------
    def select(self, preferred: list[str] | str | None, *, mask_present: bool, device: Any = None) -> AttentionBackend:
        """First preferred backend that is available, not disabled and compatible with the mask state; else sdpa."""
        if isinstance(preferred, str):
            preferred = [preferred]
        for name in preferred or []:
            if name == "sol_attn":  # BSA is layered on top by the processor, never a base backend
                continue
            backend = self._backends.get(name)
            if backend is None or name in self._disabled:
                continue
            if mask_present and backend.requires_no_mask:
                continue
            try:
                if backend.available(device):
                    return backend
            except Exception as exc:  # noqa: BLE001
                self.disable(name, f"available() raised: {exc}")
        return self._backends[DEFAULT_BACKEND]

    def run(
        self,
        backend: AttentionBackend,
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
        """Run ``backend``; on ValueError/RuntimeError disable it and fall back to sdpa for the rest of the process."""
        kwargs = dict(
            attn_mask=attn_mask, scale=scale, rope=rope, kv_heads=kv_heads,
            text_len=text_len, step_ctx=step_ctx, layout=layout,
        )
        if backend.name != DEFAULT_BACKEND and backend.name not in self._disabled:
            try:
                return backend(q, k, v, **kwargs)
            except (ValueError, RuntimeError, BackendUnavailable) as exc:
                self.disable(backend.name, f"{type(exc).__name__}: {exc}")
        return self._backends[DEFAULT_BACKEND](q, k, v, **kwargs)


_default_registry: AttentionRegistry | None = None


def get_registry() -> AttentionRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = AttentionRegistry()
    return _default_registry
