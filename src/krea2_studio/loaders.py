from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
from typing import Any, Iterable


_BLOCK_PARTS = {
    "attn.to_q.weight": "attn.wq.weight",
    "attn.to_k.weight": "attn.wk.weight",
    "attn.to_v.weight": "attn.wv.weight",
    "attn.to_gate.weight": "attn.gate.weight",
    "attn.to_out.0.weight": "attn.wo.weight",
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "ff.gate.weight": "mlp.gate.weight",
    "ff.up.weight": "mlp.up.weight",
    "ff.down.weight": "mlp.down.weight",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
}

_BASIC = {
    "img_in.weight": "first.weight",
    "img_in.bias": "first.bias",
    "time_embed.linear_1.weight": "tmlp.0.weight",
    "time_embed.linear_1.bias": "tmlp.0.bias",
    "time_embed.linear_2.weight": "tmlp.2.weight",
    "time_embed.linear_2.bias": "tmlp.2.bias",
    "time_mod_proj.weight": "tproj.1.weight",
    "time_mod_proj.bias": "tproj.1.bias",
    "txt_in.norm.weight": "txtmlp.0.scale",
    "txt_in.linear_1.weight": "txtmlp.1.weight",
    "txt_in.linear_1.bias": "txtmlp.1.bias",
    "txt_in.linear_2.weight": "txtmlp.3.weight",
    "txt_in.linear_2.bias": "txtmlp.3.bias",
    "text_fusion.projector.weight": "txtfusion.projector.weight",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.linear.weight": "last.linear.weight",
    "final_layer.linear.bias": "last.linear.bias",
    "final_layer.scale_shift_table": "last.modulation.lin",
}


def transformer_source_key(target_key: str) -> str | None:
    """Map a Diffusers Krea2 parameter to the official/Comfy checkpoint key."""
    if target_key in _BASIC:
        return _BASIC[target_key]
    match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", target_key)
    if match:
        index, part = match.groups()
        if part == "scale_shift_table":
            return f"blocks.{index}.mod.lin"
        source_part = _BLOCK_PARTS.get(part)
        return f"blocks.{index}.{source_part}" if source_part else None
    match = re.fullmatch(r"text_fusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)", target_key)
    if match:
        group, index, part = match.groups()
        source_part = _BLOCK_PARTS.get(part)
        return f"txtfusion.{group}.{index}.{source_part}" if source_part else None
    return None


def load_transformer(path: Path, device: str = "cuda"):
    """Stream the 26 GB BF16 checkpoint into a meta-initialized Diffusers model."""
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import Krea2Transformer2DModel
    from safetensors import safe_open

    with init_empty_weights(include_buffers=False):
        model = Krea2Transformer2DModel()
    expected = model.state_dict()
    missing: list[str] = []
    used: set[str] = set()
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        for target, meta_tensor in expected.items():
            source = transformer_source_key(target)
            if source is None or source not in available:
                missing.append(target)
                continue
            tensor = checkpoint.get_tensor(source)
            used.add(source)
            if target.endswith("scale_shift_table") and tensor.ndim == 1:
                tensor = tensor.reshape(meta_tensor.shape)
            if tuple(tensor.shape) != tuple(meta_tensor.shape):
                raise ValueError(f"Transformer shape mismatch: {source} {tuple(tensor.shape)} -> {target} {tuple(meta_tensor.shape)}")
            dtype = torch.float32 if "norm" in target else torch.bfloat16
            set_module_tensor_to_device(model, target, device, value=tensor, dtype=dtype)
    if missing:
        raise ValueError(f"Transformer checkpoint is incomplete; missing {len(missing)} keys: {missing[:8]}")
    unmapped = available - used
    if unmapped:
        raise ValueError(f"Transformer checkpoint has unsupported keys: {sorted(unmapped)[:8]}")
    residual = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta]
    if residual:
        raise ValueError(f"Transformer still contains meta tensors: {residual[:8]}")
    model.eval()
    return model


def resolve_snapshot(path: Path) -> Path:
    if (path / "config.json").is_file():
        return path
    snapshots = path / "snapshots"
    main_ref = path / "refs" / "main"
    if main_ref.is_file():
        referenced = snapshots / main_ref.read_text(encoding="utf-8").strip()
        if (referenced / "config.json").is_file():
            return referenced
    candidates = sorted((p for p in snapshots.glob("*") if (p / "config.json").is_file()), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No local config/tokenizer snapshot under {path}")
    return candidates[0]


def load_text_encoder(weights_path: Path, config_path: Path, device: str = "cuda"):
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel

    snapshot = resolve_snapshot(config_path)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model = Qwen3VLModel(config)
    expected = model.state_dict()
    missing: list[str] = []
    used: set[str] = set()
    with safe_open(str(weights_path), framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        for target, meta_tensor in expected.items():
            candidates = (target, f"model.{target}")
            source = next((candidate for candidate in candidates if candidate in available), None)
            if source is None:
                missing.append(target)
                continue
            tensor = checkpoint.get_tensor(source)
            used.add(source)
            if tuple(tensor.shape) != tuple(meta_tensor.shape):
                raise ValueError(f"Text encoder shape mismatch: {source} -> {target}")
            dtype = torch.bfloat16 if tensor.is_floating_point() else None
            set_module_tensor_to_device(model, target, device, value=tensor, dtype=dtype if tensor.is_floating_point() else None)
    if missing:
        raise ValueError(f"Text encoder checkpoint is incomplete; missing {len(missing)} keys: {missing[:8]}")
    if available - used:
        raise ValueError(f"Text encoder checkpoint has unsupported keys: {sorted(available - used)[:8]}")
    residual = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.is_meta]
    if residual:
        raise ValueError(f"Text encoder still contains meta tensors: {residual[:8]}")
    model.eval()
    return model, tokenizer


def load_vae(path: Path, device: str = "cuda"):
    import torch
    from diffusers import AutoencoderKLQwenImage
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
    from safetensors.torch import load_file

    state = convert_wan_vae_to_diffusers(load_file(str(path), device="cpu"))
    model = AutoencoderKLQwenImage()
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"VAE conversion was incomplete: missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return model.to(device=device, dtype=torch.bfloat16).eval()


def normalize_krea_lora_state(state: dict[str, Any]) -> dict[str, Any]:
    """Normalize Krea LoRA down/up names and fold alpha/rank into B exactly once."""
    normalized: dict[str, Any] = {}
    alpha_values: dict[str, float] = {}
    for key, tensor in state.items():
        key = _unflatten_krea_lora_key(key)
        if key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            alpha_values[base] = float(tensor.item() if hasattr(tensor, "item") else tensor)
            continue
        key = key.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight")
        normalized[key] = tensor

    a_keys = {k[: -len(".lora_A.weight")] for k in normalized if k.endswith(".lora_A.weight")}
    b_keys = {k[: -len(".lora_B.weight")] for k in normalized if k.endswith(".lora_B.weight")}
    if a_keys != b_keys or not a_keys:
        raise ValueError("LoRA must contain a complete A/B pair for every module")
    for base in a_keys:
        a_key, b_key = f"{base}.lora_A.weight", f"{base}.lora_B.weight"
        rank = normalized[a_key].shape[0]
        alpha = alpha_values.get(base)
        if alpha is not None:
            normalized[b_key] = normalized[b_key] * (alpha / rank)
    unmatched_alpha = set(alpha_values) - a_keys
    if unmatched_alpha:
        raise ValueError(f"LoRA contains alpha without weights: {sorted(unmatched_alpha)[:4]}")
    return normalized


def _unflatten_krea_lora_key(key: str) -> str:
    """Expand kohya's unambiguous lora_unet Krea module vocabulary."""
    if not key.startswith("lora_unet_"):
        return key
    flattened = key[len("lora_unet_"):]
    module, separator, suffix = flattened.partition(".")
    if not separator:
        raise ValueError(f"Malformed flattened LoRA key: {key}")
    patterns = (
        (r"blocks_(\d+)_(attn|mlp)_(wq|wk|wv|wo|gate|up|down)", r"blocks.\1.\2.\3"),
        (r"txtfusion_(layerwise_blocks|refiner_blocks)_(\d+)_(attn|mlp)_(wq|wk|wv|wo|gate|up|down)", r"txtfusion.\1.\2.\3.\4"),
        (r"last_linear", "last.linear"),
        (r"tmlp_([02])", r"tmlp.\1"),
        (r"tproj_1", "tproj.1"),
        (r"txtmlp_([13])", r"txtmlp.\1"),
        (r"txtfusion_projector", "txtfusion.projector"),
        (r"first", "first"),
    )
    dotted = next((re.sub(pattern, replacement, module) for pattern, replacement in patterns if re.fullmatch(pattern, module)), None)
    if dotted is None:
        raise ValueError(f"Unsupported flattened Krea LoRA module: {module}")
    return f"diffusion_model.{dotted}.{suffix}"


def load_lora_file(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return normalize_krea_lora_state(load_file(str(path), device="cpu"))


class PromptEmbeddingCache:
    def __init__(self, max_entries: int):
        self.max_entries = max(0, int(max_entries))
        self._items: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()

    def get(self, key: tuple[Any, ...]):
        value = self._items.pop(key, None)
        if value is not None:
            self._items[key] = value
        return value

    def put(self, key: tuple[Any, ...], value: tuple[Any, Any]) -> None:
        if self.max_entries == 0:
            return
        self._items.pop(key, None)
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
