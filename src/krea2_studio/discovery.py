from __future__ import annotations

from pathlib import Path
from typing import Any


def discover_models(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    entries = []
    seen: set[str] = set()
    for spec in config.get("models", []):
        model_id = str(spec.get("id", "")).strip()
        kind = str(spec.get("kind", ""))
        family = str(spec.get("family", ""))
        if not model_id or model_id in seen:
            raise ValueError(f"Model IDs must be non-empty and unique: {model_id!r}")
        if kind not in {"single_file_bf16", "diffusers", "unsupported_quantized"}:
            raise ValueError(f"Unsupported model kind for {model_id}: {kind}")
        if family not in {"turbo", "raw"}:
            raise ValueError(f"Model {model_id} must declare family='turbo' or family='raw'")
        seen.add(model_id)
        supported = kind != "unsupported_quantized"
        item = _model(model_id, str(spec.get("name", model_id)), str(spec.get("path", "")), kind, supported, spec.get("reason"))
        item.update({"family": family, "distilled": bool(spec.get("distilled", family == "turbo"))})
        entries.append(item)
    default_id = config["engine"]["default_model"]
    for item in entries:
        item["is_default"] = item["id"] == default_id
        if item["available"] and item["kind"] == "single_file_bf16":
            for dependency in (paths["text_encoder"], paths["vae"]):
                if not Path(dependency).is_file():
                    item["available"] = False
                    item["reason"] = f"Required component is missing: {dependency}"
                    break
    return {"items": entries, "default_id": default_id}


def get_model_spec(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for spec in config.get("models", []):
        if spec.get("id") == model_id:
            return dict(spec)
    raise ValueError(f"Unknown model: {model_id}")


def _model(model_id: str, name: str, path: str, kind: str, supported: bool, reason: str | None = None) -> dict[str, Any]:
    exists = Path(path).is_file() if kind != "diffusers" else Path(path).is_dir()
    available = exists and supported
    result = {"id": model_id, "name": name, "path": path, "kind": kind, "available": available}
    if not exists:
        result["reason"] = "Model file was not found."
    elif reason:
        result["reason"] = reason
    return result


def discover_loras(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["lora_root"])
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.rglob("*.safetensors"), key=lambda p: str(p).casefold()):
            relative = path.relative_to(root).as_posix()
            lower = relative.lower()
            category = "distillation" if "4step" in lower else ("style" if "/style/" in f"/{lower}" else "other")
            available, reason = _lora_compatibility(path)
            item = {
                "id": relative, "name": path.stem, "path": str(path), "category": category,
                "available": available,
            }
            if reason:
                item["reason"] = reason
            items.append(item)
    fast4 = next((x["id"] for x in items if x["available"] and x["category"] == "distillation" and "4step" in x["id"].lower()), None)
    return {"items": items, "fast4_lora_id": fast4}


def _lora_compatibility(path: Path) -> tuple[bool, str | None]:
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if any(".diff_b" in key for key in handle.keys()):
                return False, "Bias-delta (.diff_b) LoRA weights are not supported by the Diffusers adapter loader."
    except Exception as exc:
        return False, f"Could not inspect LoRA header: {exc}"
    return True, None


def resolve_lora(config: dict[str, Any], lora_id: str) -> Path:
    root = Path(config["paths"]["lora_root"]).resolve()
    path = (root / lora_id).resolve()
    if root not in path.parents or path.suffix.lower() != ".safetensors" or not path.is_file():
        raise ValueError(f"Unknown LoRA: {lora_id}")
    return path
