"""Model discovery: scan ``models/<kind>/`` under every configured root for ``.safetensors`` files."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel

from .config import MODEL_KINDS, AppConfig, ModelKind
from .paths import SAFE_EXT, resolve_in_roots, validate_model_name

log = logging.getLogger(__name__)

SYNC_HASH_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ModelFile:
    name: str
    kind: str
    root: str
    size: int
    hash: str | None  # sha256[:10] (A1111 "AutoV2"), None until computed

    def to_dict(self) -> dict:
        return asdict(self)


class ModelsResponse(BaseModel):
    diffusion_models: list[dict]
    text_encoders: list[dict]
    vae: list[dict]
    loras: list[dict]
    upscale_models: list[dict]
    hf_text_encoders: list[str]
    hf_vae: list[str]


class HashCache:
    """sha256[:10] cache keyed on (path, size, mtime). Hashing 26 GB is slow -> computed lazily in a thread."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._pending: set[str] = set()
        if path.is_file():
            try:
                self._data = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                self._data = {}

    def _key(self, file: Path) -> tuple[str, dict]:
        st = file.stat()
        return str(file), {"size": st.st_size, "mtime": int(st.st_mtime)}

    def get(self, file: Path) -> str | None:
        key, sig = self._key(file)
        with self._lock:
            entry = self._data.get(key)
            if entry and entry.get("size") == sig["size"] and entry.get("mtime") == sig["mtime"]:
                return entry.get("hash")
        return None

    def compute(self, file: Path) -> str:
        cached = self.get(file)
        if cached:
            return cached
        h = hashlib.sha256()
        with file.open("rb") as fh:
            for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()[:10]
        key, sig = self._key(file)
        with self._lock:
            self._data[key] = {**sig, "hash": digest}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._data, indent=1), "utf-8")
            except OSError:
                log.warning("could not persist hash cache to %s", self.path)
        return digest

    def compute_async(self, file: Path) -> None:
        key = str(file)
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)

        def _run() -> None:
            try:
                self.compute(file)
            except OSError as exc:  # pragma: no cover
                log.warning("hashing %s failed: %s", file, exc)
            finally:
                with self._lock:
                    self._pending.discard(key)

        threading.Thread(target=_run, name=f"hash-{file.name}", daemon=True).start()


class ModelRegistry:
    def __init__(self, config: AppConfig, hash_cache: HashCache | None = None):
        self.config = config
        self.hash_cache = hash_cache or HashCache(config.paths.outputs / ".hash_cache.json")

    # -- scanning ---------------------------------------------------------------------------------
    def scan(self, kind: ModelKind, background_hash: bool = False) -> list[ModelFile]:
        seen: set[str] = set()
        out: list[ModelFile] = []
        for directory in self.config.kind_dirs(kind):
            for file in sorted(directory.iterdir()):
                if not file.is_file() or file.suffix.lower() != SAFE_EXT or file.name in seen:
                    continue
                try:
                    validate_model_name(file.name)
                except ValueError:
                    continue
                seen.add(file.name)
                digest = self.hash_cache.get(file)
                if digest is None and background_hash and kind == "loras":
                    self.hash_cache.compute_async(file)
                out.append(ModelFile(file.name, kind, str(directory.parent), file.stat().st_size, digest))
        return out

    def scan_hf_dirs(self, kind: ModelKind) -> list[str]:
        """Local HF-format directories (config.json inside) under models/<kind>/."""
        names: list[str] = []
        for directory in self.config.kind_dirs(kind):
            for child in sorted(directory.iterdir()):
                if child.is_dir() and ((child / "config.json").is_file() or (child / "model_index.json").is_file()):
                    if child.name not in names:
                        names.append(child.name)
        return names

    def all_models(self) -> ModelsResponse:
        data = {kind: [m.to_dict() for m in self.scan(kind, background_hash=True)] for kind in MODEL_KINDS}
        return ModelsResponse(
            **data,
            hf_text_encoders=[self.config.defaults.text_encoder, *self.scan_hf_dirs("text_encoders")],
            hf_vae=[self.config.defaults.vae, *self.scan_hf_dirs("vae")],
        )

    # -- resolution --------------------------------------------------------------------------------
    def resolve(self, kind: ModelKind, name: str) -> Path:
        return resolve_in_roots(name, self.config.kind_dirs(kind))

    def resolve_hf(self, kind: ModelKind, name: str) -> str | Path:
        """Either a local HF directory name (under models/<kind>/) or a hub repo id."""
        if "/" not in name and "\\" not in name and ".." not in name:
            for directory in self.config.kind_dirs(kind):
                candidate = directory / name
                if candidate.is_dir():
                    return candidate.resolve(strict=True)
        if any(ch in name for ch in ("\\", "..", ":", "\0")) or name.count("/") != 1:
            raise ValueError(f"invalid HF repo id / local dir: {name!r}")
        return name

    def hash_of(self, kind: ModelKind, name: str, wait: bool = False) -> str | None:
        try:
            file = self.resolve(kind, name)
        except (FileNotFoundError, ValueError):
            return None
        if wait or file.stat().st_size <= SYNC_HASH_MAX_BYTES:
            return self.hash_cache.compute(file)
        digest = self.hash_cache.get(file)
        if digest is None:
            self.hash_cache.compute_async(file)
        return digest
