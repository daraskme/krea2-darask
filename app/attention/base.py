"""Attention backend protocol.

Tensors arrive in the layout diffusers' Krea2 processor produces: ``BSHD`` = (batch, seq, heads, head_dim), with
K/V already expanded to the Q head count (``repeat_interleave``; review.md M3). ``kv_heads`` carries the original GQA
head count so a future fused kernel (VC-Attention) can undo the expansion without the caller changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

Layout = Literal["BSHD", "BHSD"]
HEAD_DIM = 128


@dataclass
class StepContext:
    """Per-denoising-step information the processor forwards to backends (BSA scheduling, NAG)."""

    step: int
    steps: int
    sigma: float | None = None
    text_len: int = 0
    image_tokens: int = 0
    block_index: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.text_len + self.image_tokens

    @property
    def progress(self) -> float:
        return self.step / self.steps if self.steps > 0 else 1.0


@runtime_checkable
class AttentionBackend(Protocol):
    name: str
    requires_no_mask: bool

    def available(self, device: Any = None) -> bool: ...

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
    ) -> Any: ...


class BackendUnavailable(RuntimeError):
    """Raised when a backend is asked to run but cannot (import failure, wrong GPU, ineligible tensors)."""
