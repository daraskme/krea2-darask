"""Engine protocol shared by the native (diffusers) engine and the deterministic fake engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from ..schemas import GenerateRequest

# progress(step, total_steps, info) - info may contain "phase", "it_s", "vram_gb", "message"
ProgressCallback = Callable[[int, int, dict[str, Any]], None]


@dataclass
class EngineOutput:
    image: Image.Image
    seed: int
    width: int
    height: int
    elapsed_s: float
    steps: int
    attention_backend: str
    quant: str
    bsa_used: bool = False
    nag_used: bool = False
    compiled: bool = False
    gpu_name: str | None = None
    text_tokens: int | None = None
    transformer_hash: str | None = None
    lora_hashes: dict[str, str] = field(default_factory=dict)


class Engine(Protocol):
    name: str

    def capabilities(self) -> dict[str, Any]: ...

    def warmup(self, progress: ProgressCallback | None = None) -> None: ...

    def generate(self, req: GenerateRequest, seed: int, progress: ProgressCallback) -> list[EngineOutput]: ...

    def count_text_tokens(self, prompt: str) -> int: ...

    def close(self) -> None: ...


class EngineError(RuntimeError):
    pass


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    import secrets

    return secrets.randbelow(2**32)
