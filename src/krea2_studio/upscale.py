from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image


_MODEL_CACHE: tuple[Path, str, Any] | None = None
_MODEL_LOCK = Lock()


def upscale(
    image: Image.Image,
    *,
    scale: float,
    model_path: Path | None,
    device: str = "cuda",
) -> tuple[Image.Image, dict[str, Any]]:
    """Upscale a PIL image with a local Spandrel model or an honest Lanczos fallback.

    Neural upscalers commonly have a fixed native scale (for example 4x). The
    neural result is resized once to the requested final size when those scales
    differ. No model is downloaded and no ComfyUI code is imported.
    """
    requested_scale = float(scale)
    if requested_scale < 1.0:
        raise ValueError("scale must be at least 1.0")
    if image.width < 1 or image.height < 1:
        raise ValueError("image must have non-zero dimensions")

    target_size = (
        max(1, round(image.width * requested_scale)),
        max(1, round(image.height * requested_scale)),
    )
    path = Path(model_path).expanduser().resolve() if model_path is not None else None
    if path is None:
        return _lanczos(image, target_size, requested_scale, "No neural upscaler was configured.")
    if not path.is_file():
        return _lanczos(image, target_size, requested_scale, f"Upscaler model was not found: {path}")

    try:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
    except (ImportError, OSError) as exc:
        return _lanczos(image, target_size, requested_scale, f"Spandrel is unavailable: {exc}", path)

    if device.startswith("cuda") and not torch.cuda.is_available():
        return _lanczos(image, target_size, requested_scale, "CUDA is unavailable for the neural upscaler.", path)

    try:
        with _MODEL_LOCK:
            descriptor = _load_descriptor(path, device, ModelLoader, ImageModelDescriptor)
            result = _run_descriptor(image, descriptor, target_size, torch)
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            clear_upscaler_cache()
        return _lanczos(
            image,
            target_size,
            requested_scale,
            f"Neural upscaler failed: {type(exc).__name__}: {exc}",
            path,
        )

    native_scale = float(getattr(descriptor, "scale", requested_scale))
    return result, {
        "backend": "spandrel",
        "model": path.name,
        "requested_scale": requested_scale,
        "native_scale": native_scale,
        "actual_scale": result.width / image.width,
        "tiled": image.width * image.height > 1024 * 1024,
    }


def clear_upscaler_cache() -> None:
    """Release the cached descriptor when the engine needs to reclaim memory."""
    global _MODEL_CACHE
    with _MODEL_LOCK:
        had_cached_model = _MODEL_CACHE is not None
        _MODEL_CACHE = None
    if not had_cached_model:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, OSError):
        pass


def _load_descriptor(path: Path, device: str, loader_type: Any, descriptor_type: Any) -> Any:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and _MODEL_CACHE[0] == path and _MODEL_CACHE[1] == device:
        return _MODEL_CACHE[2]

    descriptor = loader_type().load_from_file(path)
    if not isinstance(descriptor, descriptor_type):
        raise TypeError(f"Expected an image-to-image model, got {type(descriptor).__name__}")
    descriptor = descriptor.to(device).eval()
    _MODEL_CACHE = (path, device, descriptor)
    return descriptor


def _run_descriptor(image: Image.Image, descriptor: Any, target_size: tuple[int, int], torch: Any) -> Image.Image:
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    source = image.convert("RGB")
    if source.width * source.height > 1024 * 1024:
        result = _run_tiled(source, descriptor, torch)
    else:
        result = _run_tile(source, descriptor, torch)
    if result.size != target_size:
        result = result.resize(target_size, Image.Resampling.LANCZOS)
    if alpha is not None and result.mode != "RGBA":
        resized_alpha = alpha.resize(target_size, Image.Resampling.LANCZOS)
        result = result.convert("RGBA")
        result.putalpha(resized_alpha)
    return result


def _run_tiled(source: Image.Image, descriptor: Any, torch: Any) -> Image.Image:
    tile_size = 512
    overlap = 32
    native_scale = int(getattr(descriptor, "scale", 0))
    if native_scale < 1:
        raise ValueError("The upscaler did not report a valid native scale")
    canvas = Image.new("RGB", (source.width * native_scale, source.height * native_scale))
    for top in range(0, source.height, tile_size):
        for left in range(0, source.width, tile_size):
            right = min(left + tile_size, source.width)
            bottom = min(top + tile_size, source.height)
            padded = (
                max(0, left - overlap),
                max(0, top - overlap),
                min(source.width, right + overlap),
                min(source.height, bottom + overlap),
            )
            tile = _run_tile(source.crop(padded), descriptor, torch)
            crop = (
                (left - padded[0]) * native_scale,
                (top - padded[1]) * native_scale,
                (right - padded[0]) * native_scale,
                (bottom - padded[1]) * native_scale,
            )
            canvas.paste(tile.crop(crop), (left * native_scale, top * native_scale))
    return canvas


def _run_tile(source: Image.Image, descriptor: Any, torch: Any) -> Image.Image:
    raw = bytearray(source.tobytes())
    tensor = torch.frombuffer(raw, dtype=torch.uint8)
    tensor = tensor.reshape(source.height, source.width, 3).permute(2, 0, 1).unsqueeze(0)
    model_dtype = getattr(descriptor, "dtype", torch.float32)
    tensor = tensor.to(device=descriptor.device, dtype=model_dtype).div_(255)
    with torch.inference_mode():
        output = descriptor(tensor)
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] not in {1, 3, 4}:
        raise ValueError(f"Unexpected upscaler output shape: {tuple(output.shape)}")
    output = output[0].detach().float().clamp_(0, 1).mul_(255).round().to(torch.uint8).cpu()
    channels = int(output.shape[0])
    mode = {1: "L", 3: "RGB", 4: "RGBA"}[channels]
    output = output.permute(1, 2, 0).contiguous()
    if channels == 1:
        output = output.squeeze(-1)
    return Image.fromarray(output.numpy(), mode=mode)


def _lanczos(
    image: Image.Image,
    target_size: tuple[int, int],
    requested_scale: float,
    reason: str,
    model_path: Path | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "backend": "pillow_lanczos",
        "reason": reason,
        "requested_scale": requested_scale,
        "actual_scale": target_size[0] / image.width,
        "tiled": False,
    }
    if model_path is not None:
        metadata["model"] = model_path.name
    return image.resize(target_size, Image.Resampling.LANCZOS), metadata
