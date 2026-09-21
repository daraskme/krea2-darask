"""Path validation for user-supplied model / output names (review.md M7 / G9).

Rules:
* model files: ``.safetensors`` only, root-direct relative name, no separators / ``..`` / ``:`` / NUL
* every resolved path must be inside one of the configured roots (``resolve(strict=True)`` + ``is_relative_to``)
"""

from __future__ import annotations

import re
from pathlib import Path

SAFE_EXT = ".safetensors"
_FORBIDDEN_CHARS = ("/", "\\", ":", "\0")
_OUTPUT_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/\d{6}_\d+_[0-9a-f]{6}\.(png|jpg|webp)$")


class UnsafePathError(ValueError):
    pass


def validate_model_name(name: str) -> str:
    """Return ``name`` if it is an acceptable root-direct ``.safetensors`` file name; raise otherwise."""
    if not isinstance(name, str) or not name:
        raise UnsafePathError("empty model name")
    if len(name) > 255:
        raise UnsafePathError("model name too long")
    if name in (".", "..") or ".." in name.split("."):
        raise UnsafePathError("'..' is not allowed in model names")
    if any(ch in name for ch in _FORBIDDEN_CHARS):
        raise UnsafePathError(f"path separators / drive letters are not allowed: {name!r}")
    if any(ord(ch) < 32 for ch in name):
        raise UnsafePathError("control characters are not allowed")
    if name.startswith(".") or name.startswith("~"):
        raise UnsafePathError("hidden / home-relative names are not allowed")
    if not name.lower().endswith(SAFE_EXT):
        raise UnsafePathError(f"only {SAFE_EXT} files are allowed (got {name!r})")
    return name


def resolve_in_roots(name: str, roots: list[Path]) -> Path:
    """Resolve a validated model file name inside the first root that contains it."""
    validate_model_name(name)
    for root in roots:
        try:
            root_resolved = root.resolve(strict=True)
            candidate = (root_resolved / name).resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if not candidate.is_relative_to(root_resolved):
            raise UnsafePathError(f"{name!r} escapes its root")
        if candidate.parent != root_resolved:
            raise UnsafePathError(f"{name!r} must be directly under the model root")
        if not candidate.is_file():
            continue
        return candidate
    raise FileNotFoundError(f"{name!r} not found in {[str(r) for r in roots]}")


def validate_output_id(output_id: str) -> str:
    """Output ids are server-generated: ``YYYY-MM-DD/HHMMSS_<seed>_<hex6>.<ext>``."""
    if not isinstance(output_id, str) or not _OUTPUT_ID_RE.match(output_id):
        raise UnsafePathError(f"invalid output id {output_id!r}")
    return output_id


def resolve_output(output_id: str, outputs_root: Path) -> Path:
    validate_output_id(output_id)
    root = outputs_root.resolve(strict=True)
    candidate = (root / output_id).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise UnsafePathError("output escapes outputs root")
    return candidate
