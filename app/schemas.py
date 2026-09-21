"""Request / result schemas shared by the API, the engines and the PNG metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from . import __version__
from .presets import (
    BSA_MIN_TOKENS,
    MAX_TEXT_TOKENS,
    bsa_token_eligible,
    get_workflow,
    hires_edge_warning,
    image_tokens,
    round_down_16,
)

OutputFormat = Literal["png", "jpg", "webp"]
MAX_PROMPT_CHARS = 20_000
MAX_LORAS = 16


class RequestValidationError(ValueError):
    """Raised by GenerateRequest.validate(); the API turns it into HTTP 400."""


class LoraSpec(BaseModel):
    name: str = Field(max_length=255)
    strength: float = Field(default=1.0, ge=-4.0, le=4.0)
    enabled: bool = True
    hash: str | None = Field(default=None, max_length=64)


class NagParams(BaseModel):
    enabled: bool = False
    negative: str = Field(default="", max_length=MAX_PROMPT_CHARS)
    phi: float = Field(default=4.0, ge=0.0, le=20.0)
    tau: float = Field(default=2.5, ge=1.0, le=20.0)
    alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    sigma_start: float = Field(default=1000.0, ge=0.0)
    sigma_end: float = Field(default=0.0, ge=0.0)


class HiresParams(BaseModel):
    enabled: bool = False
    upscaler: str | None = Field(default=None, max_length=255)
    scale: float = Field(default=2.0, gt=1.0, le=4.0)
    denoise: float = Field(default=0.35, gt=0.0, le=1.0)
    steps: int = Field(default=8, ge=1, le=100)
    source_output_id: str | None = Field(default=None, max_length=128)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", max_length=MAX_PROMPT_CHARS)
    negative_prompt: str = Field(default="", max_length=MAX_PROMPT_CHARS)
    preset: str = Field(default="fast_4step", max_length=64)
    width: int = Field(default=832, ge=16, le=4096)
    height: int = Field(default=1216, ge=16, le=4096)
    steps: int = Field(default=4, ge=1, le=200)
    cfg: float = Field(default=1.0, ge=0.0, le=30.0)  # ComfyUI cfg; engine converts to guidance = cfg - 1
    sampler: str = Field(default="euler", max_length=32)
    scheduler: str = Field(default="simple", max_length=32)
    denoise: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    batch_size: int = Field(default=1, ge=1, le=16)
    transformer: str = Field(default="", max_length=255)
    text_encoder: str = Field(default="", max_length=255)
    vae: str = Field(default="", max_length=255)
    loras: list[LoraSpec] = Field(default_factory=list, max_length=MAX_LORAS)
    lora_4step: bool = True
    lora_4step_name: str | None = Field(default=None, max_length=255)
    bsa: bool | None = None  # None = auto (token threshold), True = force request, False = off
    bsa_tau: float = Field(default=1.3, ge=0.0, le=4.0)
    bsa_start_percent: float = Field(default=0.2, ge=0.0, le=1.0)
    nag: NagParams = Field(default_factory=NagParams)
    hires: HiresParams = Field(default_factory=HiresParams)
    attention: str | None = Field(default=None, max_length=32)
    compact_text_tokens: bool | None = None
    output_format: OutputFormat = "png"
    text_tokens: int | None = Field(default=None, ge=0, le=MAX_TEXT_TOKENS)

    @field_validator("width", "height")
    @classmethod
    def _round16(cls, value: int) -> int:
        return round_down_16(value)

    @model_validator(mode="after")
    def _preset_defaults(self) -> GenerateRequest:
        wf = get_workflow(self.preset)
        if wf is not None and wf.hires:
            self.hires.enabled = True
            self.lora_4step = False
        return self

    # -- derived ---------------------------------------------------------------------------------
    @property
    def guidance_scale(self) -> float:
        return max(0.0, self.cfg - 1.0)

    def active_loras(self) -> list[LoraSpec]:
        return [lora for lora in self.loras if lora.enabled and lora.strength != 0.0]

    def effective_text_tokens(self) -> int:
        return MAX_TEXT_TOKENS if self.text_tokens is None else self.text_tokens

    def sampling_size(self) -> tuple[int, int]:
        """Size of the denoised latent (after hires upscale when enabled)."""
        if self.hires.enabled:
            return round_down_16(int(self.width * self.hires.scale)), round_down_16(int(self.height * self.hires.scale))
        return self.width, self.height

    def bsa_effective(self, min_tokens: int = BSA_MIN_TOKENS) -> bool:
        """Final BSA decision: explicit False wins, explicit True requires the token threshold, None = auto."""
        if self.bsa is False:
            return False
        w, h = self.sampling_size()
        return bsa_token_eligible(w, h, self.effective_text_tokens(), min_tokens)

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.text_tokens is not None and self.text_tokens >= MAX_TEXT_TOKENS:
            out.append(f"prompt reaches the {MAX_TEXT_TOKENS}-token limit and will be truncated")
        if self.hires.enabled:
            warn = hires_edge_warning(self.width, self.height, self.hires.scale)
            if warn:
                out.append(warn)
        return out

    def validate(self) -> GenerateRequest:
        """Cross-field rules (bench-derived, review.md C4/M6). Raises RequestValidationError."""
        if self.lora_4step and self.bsa is True:
            raise RequestValidationError("4-step LoRA and Block Sparse Attention cannot be combined")
        if self.lora_4step and self.bsa is None and self.bsa_effective():
            raise RequestValidationError(
                "4-step LoRA and Block Sparse Attention cannot be combined: this size reaches the "
                f"{BSA_MIN_TOKENS}-token threshold; set bsa=false or disable the 4-step LoRA"
            )
        if self.hires.enabled and self.lora_4step:
            raise RequestValidationError("Hires pass must not use the 4-step LoRA")
        if self.nag.enabled and self.cfg != 1.0:
            raise RequestValidationError("NAG requires cfg 1.0 (positive branch only)")
        if self.nag.enabled and self.batch_size != 1:
            raise RequestValidationError("NAG supports batch_size 1 only")
        if self.hires.enabled and not self.hires.source_output_id and not self.prompt:
            raise RequestValidationError("Hires needs a source image or a prompt")
        if image_tokens(self.width, self.height) < 1:
            raise RequestValidationError("image too small")
        return self


class GenerateResult(BaseModel):
    """What the engine reports back per image (stored in the krea2gui PNG chunk)."""

    output_id: str
    seed: int
    width: int
    height: int
    elapsed_s: float
    steps: int
    engine: str
    attention_backend: str
    quant: str
    bsa_used: bool
    nag_used: bool
    compiled: bool = False
    gpu_name: str | None = None
    text_tokens: int | None = None
    transformer_hash: str | None = None
    lora_hashes: dict[str, str] = Field(default_factory=dict)
    version: str = __version__


class Krea2GuiMetadata(BaseModel):
    """The full ``krea2gui`` JSON chunk. Bounded and schema-validated on read (G10)."""

    schema_version: int = 1
    app: str = "krea2-darask"
    version: str = __version__
    request: GenerateRequest
    result: GenerateResult
