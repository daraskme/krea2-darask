"""Resolution / workflow presets and the single token-count function (review.md M4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

PATCH_PIXELS = 16  # VAE f8 * patch 2
MAX_TEXT_TOKENS = 512
BSA_MIN_TOKENS = 12288
MAX_TRAINED_EDGE = 2048


@dataclass(frozen=True)
class ResolutionPreset:
    id: str
    name: str
    width: int
    height: int
    note: str = ""

    @property
    def image_tokens(self) -> int:
        return image_tokens(self.width, self.height)

    def to_dict(self) -> dict:
        return {**asdict(self), "image_tokens": self.image_tokens}


# Unique 10 (+ swap button in the UI) + 832x2048 from the user's real workflow. Portrait first.
RESOLUTION_PRESETS: tuple[ResolutionPreset, ...] = (
    ResolutionPreset("512x2048", "Tall 1:4", 512, 2048),
    ResolutionPreset("576x1728", "Tall 1:3", 576, 1728),
    ResolutionPreset("832x2048", "Tall 832 (workflow)", 832, 2048, "hires 2x -> 4096 edge is outside the trained range"),
    ResolutionPreset("768x1344", "Portrait 9:16", 768, 1344),
    ResolutionPreset("832x1216", "Portrait 2:3", 832, 1216),
    ResolutionPreset("896x1152", "Portrait 3:4", 896, 1152),
    ResolutionPreset("1024x1024", "Square", 1024, 1024),
    ResolutionPreset("1152x896", "Landscape 4:3", 1152, 896),
    ResolutionPreset("1216x832", "Landscape 3:2", 1216, 832),
    ResolutionPreset("1344x768", "Landscape 16:9", 1344, 768),
    ResolutionPreset("1536x640", "Ultrawide", 1536, 640),
)


@dataclass(frozen=True)
class WorkflowPreset:
    id: str
    name: str
    steps: int
    cfg: float  # ComfyUI cfg; engine boundary converts to guidance_scale = cfg - 1
    lora_4step: bool
    sampler: str = "euler"
    scheduler: str = "simple"
    denoise: float = 1.0
    hires: bool = False
    bsa: bool = False
    raw_model: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


WORKFLOW_PRESETS: tuple[WorkflowPreset, ...] = (
    WorkflowPreset(
        "fast_4step", "Fast 4-step", 4, 1.0, True,
        description="Turbo + 4-step LoRA(1.0), euler/simple, cfg 1.0 (Krea2_T2I_Fast4step_darask)",
    ),
    WorkflowPreset("turbo_8step", "Turbo 8-step", 8, 1.0, False, description="Turbo without LoRA, 8 steps"),
    WorkflowPreset(
        "hires_2x", "Hires 2x", 8, 1.0, False, denoise=0.35, hires=True, bsa=True,
        description="4x-UltraSharpV2 -> x0.5 -> VAE encode -> BSA(sol_attn tau 1.3) -> 8 steps denoise 0.35",
    ),
    WorkflowPreset(
        "raw_52step", "Raw 52-step", 52, 3.5, False, raw_model=True,
        description="Raw checkpoint, cfg 3.5 (guidance 2.5), 52 steps",
    ),
)


def round_down_16(value: int) -> int:
    return max(PATCH_PIXELS, (int(value) // PATCH_PIXELS) * PATCH_PIXELS)


def image_tokens(width: int, height: int) -> int:
    """Latent token count of a WxH image: (W//16)*(H//16)."""
    return (width // PATCH_PIXELS) * (height // PATCH_PIXELS)


def total_tokens(width: int, height: int, text_tokens: int) -> int:
    """The one place where 'text + image tokens' is computed (BSA eligibility, memory estimates)."""
    return image_tokens(width, height) + int(text_tokens)


def bsa_token_eligible(width: int, height: int, text_tokens: int, min_tokens: int = BSA_MIN_TOKENS) -> bool:
    return total_tokens(width, height, text_tokens) >= min_tokens


def hires_size(width: int, height: int, scale: float = 2.0) -> tuple[int, int]:
    return round_down_16(int(width * scale)), round_down_16(int(height * scale))


def hires_edge_warning(width: int, height: int, scale: float = 2.0) -> str | None:
    w, h = hires_size(width, height, scale)
    if max(w, h) > MAX_TRAINED_EDGE:
        return f"hires output {w}x{h} exceeds the trained range (edge > {MAX_TRAINED_EDGE})"
    return None


def cfg_to_guidance(cfg: float) -> float:
    """ComfyUI cfg -> diffusers Krea2 guidance_scale (cond + g*(cond-uncond)); cfg 1.0 -> 0.0 (no CFG)."""
    return max(0.0, float(cfg) - 1.0)


def denoise_steps(steps: int, denoise: float) -> int:
    """ComfyUI KSampler: total schedule length when denoise < 1 (the last `steps` entries are sampled)."""
    if denoise >= 1.0:
        return steps
    if denoise <= 0.0:
        raise ValueError("denoise must be > 0")
    return int(steps / denoise)


def get_resolution(preset_id: str) -> ResolutionPreset | None:
    return next((p for p in RESOLUTION_PRESETS if p.id == preset_id), None)


def get_workflow(preset_id: str) -> WorkflowPreset | None:
    return next((p for p in WORKFLOW_PRESETS if p.id == preset_id), None)


def presets_payload() -> dict:
    return {
        "resolutions": [p.to_dict() for p in RESOLUTION_PRESETS],
        "workflows": [p.to_dict() for p in WORKFLOW_PRESETS],
        "limits": {
            "max_text_tokens": MAX_TEXT_TOKENS,
            "bsa_min_tokens": BSA_MIN_TOKENS,
            "max_trained_edge": MAX_TRAINED_EDGE,
            "patch_pixels": PATCH_PIXELS,
        },
    }
