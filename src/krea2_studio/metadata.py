from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags, PngImagePlugin


PNG_METADATA_KEY = "krea2_studio"
EXIF_USER_COMMENT = 0x9286
UNICODE_PREFIX = b"UNICODE\x00"
UNICODE_BOM = b"\xfe\xff"


def _payload(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def encode_user_comment(text: str) -> bytes:
    return UNICODE_PREFIX + UNICODE_BOM + text.encode("utf-16-be")


def decode_user_comment(value: bytes | str) -> str:
    if isinstance(value, str):
        return value
    if value.startswith(UNICODE_PREFIX):
        body = value[len(UNICODE_PREFIX):]
        if body.startswith((b"\xfe\xff", b"\xff\xfe")):
            return body.decode("utf-16")
        return body.decode("utf-16-be")
    return value.decode("utf-8", errors="replace")


def save_image(image: Image.Image, output_root: Path, stem: str, metadata: dict[str, Any]) -> tuple[Path, Path]:
    folder = output_root / datetime.now().strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    png_path = folder / f"{stem}.png"
    json_path = folder / f"{stem}.json"
    text = _payload(metadata)
    exif = Image.Exif()
    exif[ExifTags.IFD.Exif] = {EXIF_USER_COMMENT: encode_user_comment(text)}
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_itxt(PNG_METADATA_KEY, text, lang="", tkey="Krea 2 Studio metadata")
    image.save(png_path, format="PNG", pnginfo=pnginfo, exif=exif)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return png_path, json_path


def read_image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        if PNG_METADATA_KEY in image.info:
            return json.loads(image.info[PNG_METADATA_KEY])
        comment = image.getexif().get_ifd(ExifTags.IFD.Exif).get(EXIF_USER_COMMENT)
        if comment:
            return json.loads(decode_user_comment(comment))
    sidecar = path.with_suffix(".json")
    return json.loads(sidecar.read_text(encoding="utf-8"))
