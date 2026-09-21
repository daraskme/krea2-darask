"""outputs/YYYY-MM-DD/HHMMSS_<seed>_<hex6>.<ext> - saving (with metadata) and gallery listing."""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .metadata import read_image, write_image
from .paths import resolve_output, validate_output_id
from .schemas import Krea2GuiMetadata

_EXT = {"png": "png", "jpg": "jpg", "webp": "webp"}
THUMB_SIZE = 384


@dataclass
class OutputEntry:
    id: str
    width: int
    height: int
    mtime: float
    size: int

    def to_dict(self) -> dict:
        return {"id": self.id, "width": self.width, "height": self.height, "mtime": self.mtime, "size": self.size}


class OutputStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def new_id(self, seed: int, fmt: str, now: datetime | None = None) -> str:
        now = now or datetime.now()
        return f"{now:%Y-%m-%d}/{now:%H%M%S}_{seed}_{secrets.token_hex(3)}.{_EXT[fmt]}"

    def save(self, image: Image.Image, meta: Krea2GuiMetadata, fmt: str = "png") -> str:
        output_id = meta.result.output_id
        validate_output_id(output_id)
        path = self.root / output_id
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_image(image, path, meta, fmt=fmt)
        return output_id

    def path_of(self, output_id: str) -> Path:
        return resolve_output(output_id, self.root)

    def thumbnail(self, output_id: str) -> Path:
        src = self.path_of(output_id)
        thumb_dir = self.root / ".thumbs" / src.parent.name
        thumb = thumb_dir / (src.stem + ".webp")
        if thumb.is_file() and thumb.stat().st_mtime >= src.stat().st_mtime:
            return thumb
        thumb_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.convert("RGB").save(thumb, format="WEBP", quality=80)
        return thumb

    def list(self, limit: int = 200, offset: int = 0) -> list[OutputEntry]:
        entries: list[OutputEntry] = []
        for day in sorted((d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")), reverse=True):
            for file in day.iterdir():
                if not file.is_file() or file.suffix.lower() not in (".png", ".jpg", ".webp"):
                    continue
                output_id = f"{day.name}/{file.name}"
                try:
                    validate_output_id(output_id)
                except ValueError:
                    continue
                st = file.stat()
                try:
                    with Image.open(file) as img:
                        w, h = img.size
                except OSError:
                    continue
                entries.append(OutputEntry(output_id, w, h, st.st_mtime, st.st_size))
        entries.sort(key=lambda e: e.mtime, reverse=True)
        return entries[offset : offset + limit]

    def metadata(self, output_id: str) -> dict:
        path = self.path_of(output_id)
        result = read_image(path)
        return {
            "id": output_id,
            "source": result.source,
            "request": result.request,
            "parameters": result.parameters,
            "krea2gui": result.krea2gui.model_dump(exclude_none=True) if result.krea2gui else None,
            "warnings": result.warnings,
        }

    def delete(self, output_id: str) -> None:
        path = self.path_of(output_id)
        path.unlink()
        thumb = self.root / ".thumbs" / path.parent.name / (path.stem + ".webp")
        if thumb.is_file():
            thumb.unlink()
