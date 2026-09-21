"""PNG / JPEG / WebP metadata (review.md G10, M5).

Written:
* ``parameters``  - A1111-compatible text (prompt / "Negative prompt:" / key: value line with >= 3 pairs)
* ``krea2gui``    - full pydantic-validated JSON (Krea2GuiMetadata)
Read (in this order): ``parameters`` -> ``krea2gui`` -> best-effort ComfyUI ``prompt`` / ``workflow`` JSON.
No fake ComfyUI ``prompt`` JSON is synthesised.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import piexif
import piexif.helper
from PIL import Image, PngImagePlugin

from . import __version__
from .schemas import GenerateRequest, GenerateResult, Krea2GuiMetadata, LoraSpec

PNG_TEXT_LIMIT = 1024 * 1024  # 1 MiB total for our text chunks
JPEG_APP1_LIMIT = 65_533  # bytes available for the whole EXIF payload in one APP1 segment
KREA2GUI_JSON_LIMIT = 256 * 1024
_RE_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')


# ---------------------------------------------------------------------------------------------------
# A1111 helpers
# ---------------------------------------------------------------------------------------------------
def quote(text: Any) -> str:
    """A1111 ``infotext_utils.quote``: values containing , : " or newlines are JSON-quoted."""
    text = str(text)
    if "," not in text and "\n" not in text and ":" not in text and '"' not in text:
        return text
    return json.dumps(text, ensure_ascii=False)


def unquote(text: str) -> str:
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def build_parameters(req: GenerateRequest, res: GenerateResult) -> str:
    """A1111 infotext. Last line holds >= 3 comma separated ``Key: value`` pairs (quote-compatible)."""
    lines = [req.prompt.strip()]
    if req.negative_prompt.strip():
        lines.append(f"Negative prompt: {req.negative_prompt.strip()}")
    pairs: list[tuple[str, Any]] = [
        ("Steps", res.steps),
        ("Sampler", req.sampler.capitalize()),
        ("Schedule type", req.scheduler.capitalize()),
        ("CFG scale", req.cfg),
        ("Seed", res.seed),
        ("Size", f"{res.width}x{res.height}"),
        ("Model", Path(req.transformer).stem if req.transformer else "krea2"),
    ]
    if res.transformer_hash:
        pairs.append(("Model hash", res.transformer_hash))
    if req.hires.enabled:
        pairs.append(("Denoising strength", req.hires.denoise))
        pairs.append(("Hires upscale", req.hires.scale))
        if req.hires.upscaler:
            pairs.append(("Hires upscaler", Path(req.hires.upscaler).stem))
    loras = [lora for lora in req.active_loras()]
    if req.lora_4step and req.lora_4step_name:
        loras = [LoraSpec(name=req.lora_4step_name, strength=1.0, hash=res.lora_hashes.get(req.lora_4step_name)), *loras]
    if loras:
        hashes = ", ".join(
            f"{Path(lora.name).stem}: {lora.hash or res.lora_hashes.get(lora.name) or 'unknown'}" for lora in loras
        )
        pairs.append(("Lora hashes", hashes))
    if req.nag.enabled:
        pairs.append(("NAG", f"phi {req.nag.phi} tau {req.nag.tau} alpha {req.nag.alpha}"))
    if res.bsa_used:
        pairs.append(("BSA", f"sol_attn tau {req.bsa_tau}"))
    pairs.append(("Version", f"krea2-darask {__version__}"))
    lines.append(", ".join(f"{k}: {quote(v)}" for k, v in pairs))
    return "\n".join(lines)


@dataclass
class ParsedParameters:
    prompt: str = ""
    negative_prompt: str = ""
    settings: dict[str, str] = field(default_factory=dict)


def parse_parameters(text: str) -> ParsedParameters:
    """Inverse of build_parameters (A1111 ``parse_generation_parameters`` semantics)."""
    out = ParsedParameters()
    lines = text.strip().split("\n")
    if not lines:
        return out
    last = lines[-1]
    pairs = _RE_PARAM.findall(last)
    has_settings = len(pairs) >= 3
    body = lines[:-1] if has_settings else lines
    done_prompt = False
    prompt_lines: list[str] = []
    neg_lines: list[str] = []
    for line in body:
        if line.startswith("Negative prompt:"):
            done_prompt = True
            line = line[len("Negative prompt:") :].strip()
        (neg_lines if done_prompt else prompt_lines).append(line)
    out.prompt = "\n".join(prompt_lines).strip()
    out.negative_prompt = "\n".join(neg_lines).strip()
    if has_settings:
        for key, value in pairs:
            out.settings[key.strip()] = unquote(value.strip())
    return out


def parameters_to_request(parsed: ParsedParameters) -> dict:
    """Best-effort GenerateRequest fields from an A1111 infotext (used when no krea2gui chunk exists)."""
    s = parsed.settings
    req: dict[str, Any] = {"prompt": parsed.prompt, "negative_prompt": parsed.negative_prompt}
    if "Steps" in s and s["Steps"].isdigit():
        req["steps"] = int(s["Steps"])
    if "Seed" in s and s["Seed"].isdigit():
        req["seed"] = int(s["Seed"])
    if "CFG scale" in s:
        try:
            req["cfg"] = float(s["CFG scale"])
        except ValueError:
            pass
    if "Size" in s and "x" in s["Size"]:
        try:
            w, h = s["Size"].lower().split("x")
            req["width"], req["height"] = int(w), int(h)
        except ValueError:
            pass
    if "Sampler" in s:
        req["sampler"] = s["Sampler"].lower()
    if "Schedule type" in s:
        req["scheduler"] = s["Schedule type"].lower()
    if "Model" in s and s["Model"]:
        req["transformer"] = s["Model"] + ("" if s["Model"].endswith(".safetensors") else ".safetensors")
    loras: list[dict] = []
    if "Lora hashes" in s:
        for item in s["Lora hashes"].split(","):
            if ":" in item:
                name, digest = item.rsplit(":", 1)
                loras.append({"name": name.strip() + ".safetensors", "strength": 1.0, "hash": digest.strip()})
    if loras:
        req["loras"] = loras
    return req


# ---------------------------------------------------------------------------------------------------
# ComfyUI best-effort reader (no writer!)
# ---------------------------------------------------------------------------------------------------
def comfy_prompt_to_request(prompt_json: dict) -> dict:
    """Extract prompt / seed / steps / cfg / size / LoRAs from a ComfyUI API-format ``prompt`` JSON."""
    req: dict[str, Any] = {}
    loras: list[dict] = []
    texts: list[str] = []
    for node in prompt_json.values():
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {}) if isinstance(node.get("inputs"), dict) else {}
        if ctype == "CLIPTextEncode" and isinstance(inputs.get("text"), str):
            texts.append(inputs["text"])
        elif ctype in ("KSampler", "KSamplerAdvanced"):
            for src, dst in (("seed", "seed"), ("noise_seed", "seed"), ("steps", "steps"), ("cfg", "cfg")):
                if isinstance(inputs.get(src), int | float):
                    req[dst] = inputs[src]
            if isinstance(inputs.get("sampler_name"), str):
                req["sampler"] = inputs["sampler_name"]
            if isinstance(inputs.get("scheduler"), str):
                req["scheduler"] = inputs["scheduler"]
            if isinstance(inputs.get("denoise"), int | float):
                req["denoise"] = float(inputs["denoise"])
        elif ctype in ("EmptyLatentImage", "EmptySD3LatentImage"):
            if isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
                req["width"], req["height"] = inputs["width"], inputs["height"]
        elif ctype in ("LoraLoaderModelOnly", "LoraLoader") and isinstance(inputs.get("lora_name"), str):
            loras.append({"name": inputs["lora_name"], "strength": float(inputs.get("strength_model", 1.0))})
        elif ctype == "UNETLoader" and isinstance(inputs.get("unet_name"), str):
            req["transformer"] = inputs["unet_name"]
    if texts:
        req["prompt"] = texts[0]
        if len(texts) > 1:
            req["negative_prompt"] = texts[1]
    if loras:
        req["loras"] = loras
    if "steps" in req:
        req["steps"] = int(req["steps"])
    if "seed" in req:
        req["seed"] = int(req["seed"])
    return req


# ---------------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------------
def _krea2gui_json(meta: Krea2GuiMetadata) -> str:
    text = meta.model_dump_json(exclude_none=True)
    if len(text.encode("utf-8")) > KREA2GUI_JSON_LIMIT:
        raise ValueError("krea2gui metadata exceeds size limit")
    return text


def write_image(image: Image.Image, path: Path, meta: Krea2GuiMetadata, fmt: str = "png", quality: int = 95) -> None:
    parameters = build_parameters(meta.request, meta.result)
    gui_json = _krea2gui_json(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        total = len(parameters.encode("utf-8")) + len(gui_json.encode("utf-8"))
        if total > PNG_TEXT_LIMIT:
            raise ValueError(f"PNG text metadata {total} bytes exceeds the 1 MiB limit")
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", parameters)
        info.add_text("krea2gui", gui_json)
        image.save(path, format="PNG", pnginfo=info, compress_level=1)
        return
    exif_bytes = build_exif(parameters, gui_json, meta.request.prompt)
    if fmt == "jpg":
        image.convert("RGB").save(path, format="JPEG", quality=quality, exif=exif_bytes)
    elif fmt == "webp":
        image.save(path, format="WEBP", quality=quality, exif=exif_bytes)
    else:
        raise ValueError(f"unsupported format {fmt}")


def build_exif(parameters: str, gui_json: str | None, prompt: str) -> bytes:
    """EXIF: UserComment = parameters (A1111 identical), XPComment = krea2gui JSON, ImageDescription = prompt.
    Shrinks to fit the 65 533 byte APP1 limit (drop krea2gui first, then the description)."""

    def _dump(with_gui: bool, with_desc: bool) -> bytes:
        zeroth: dict[int, Any] = {piexif.ImageIFD.Software: f"krea2-darask {__version__}"}
        if with_desc and prompt:
            zeroth[piexif.ImageIFD.ImageDescription] = prompt.encode("utf-8")[:4000]
        if with_gui and gui_json:
            zeroth[piexif.ImageIFD.XPComment] = gui_json.encode("utf-16le") + b"\x00\x00"
        exif_ifd = {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters, encoding="unicode")}
        return piexif.dump({"0th": zeroth, "Exif": exif_ifd})

    for with_gui, with_desc in ((True, True), (False, True), (False, False)):
        data = _dump(with_gui, with_desc)
        if len(data) <= JPEG_APP1_LIMIT:
            return data
    raise ValueError("parameters text alone exceeds the EXIF APP1 limit")


# ---------------------------------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------------------------------
@dataclass
class ReadResult:
    source: str  # "krea2gui" | "parameters" | "comfyui" | "none"
    request: dict[str, Any]
    parameters: str | None = None
    krea2gui: Krea2GuiMetadata | None = None
    comfy_prompt: dict | None = None
    warnings: list[str] = field(default_factory=list)


def _read_exif_texts(image: Image.Image) -> tuple[str | None, str | None]:
    raw = image.info.get("exif")
    if not raw:
        return None, None
    try:
        exif = piexif.load(raw)
    except Exception:  # noqa: BLE001 - malformed EXIF from arbitrary user files
        return None, None
    parameters = None
    gui = None
    uc = exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
    if uc:
        try:
            parameters = piexif.helper.UserComment.load(uc)
        except ValueError:
            parameters = None
    xp = exif.get("0th", {}).get(piexif.ImageIFD.XPComment)
    if xp:
        try:
            gui = bytes(xp).decode("utf-16le").rstrip("\x00")
        except UnicodeDecodeError:
            gui = None
    return parameters, gui


def _extract_texts(image: Image.Image) -> tuple[Any, Any, Any, Any]:
    info = dict(image.info)
    parameters = info.get("parameters")
    gui_text = info.get("krea2gui")
    if parameters is None and gui_text is None:
        parameters, gui_text = _read_exif_texts(image)
    return parameters, gui_text, info.get("prompt"), info.get("workflow")


def read_image(source_image: Path | Image.Image) -> ReadResult:
    if isinstance(source_image, Image.Image):
        parameters, gui_text, comfy_prompt, comfy_workflow = _extract_texts(source_image)
    else:
        with Image.open(source_image) as image:
            image.load()
            parameters, gui_text, comfy_prompt, comfy_workflow = _extract_texts(image)

    warnings: list[str] = []
    request: dict[str, Any] = {}
    parsed_gui: Krea2GuiMetadata | None = None
    source = "none"

    if isinstance(parameters, str) and parameters.strip():
        request.update(parameters_to_request(parse_parameters(parameters)))
        source = "parameters"

    if isinstance(gui_text, str) and gui_text:
        if len(gui_text.encode("utf-8")) > KREA2GUI_JSON_LIMIT:
            warnings.append("krea2gui chunk too large, ignored")
        else:
            try:
                parsed_gui = Krea2GuiMetadata.model_validate_json(gui_text)
                request = parsed_gui.request.model_dump(exclude_none=True)
                request["seed"] = parsed_gui.result.seed
                source = "krea2gui"
            except ValueError as exc:
                warnings.append(f"krea2gui chunk invalid: {str(exc)[:200]}")

    comfy_json: dict | None = None
    if source == "none":
        for candidate in (comfy_prompt, comfy_workflow):
            if isinstance(candidate, str) and len(candidate) < KREA2GUI_JSON_LIMIT * 4:
                try:
                    data = json.loads(candidate)
                except ValueError:
                    continue
                if isinstance(data, dict) and any(isinstance(v, dict) and "class_type" in v for v in data.values()):
                    comfy_json = data
                    request.update(comfy_prompt_to_request(data))
                    source = "comfyui"
                    break

    return ReadResult(
        source=source,
        request=request,
        parameters=parameters if isinstance(parameters, str) else None,
        krea2gui=parsed_gui,
        comfy_prompt=comfy_json,
        warnings=warnings,
    )
