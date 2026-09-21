"""FastAPI application: 127.0.0.1-only, Host-header checked, same-origin static GUI (web/dist) in production."""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .config import REPO_ROOT, AppConfig, load_config
from .engine.base import Engine
from .metadata import read_image
from .models import ModelRegistry
from .outputs import OutputStore
from .paths import UnsafePathError, validate_output_id
from .presets import hires_edge_warning, presets_payload, total_tokens
from .queue import EventBus, GenerationQueue
from .schemas import GenerateRequest, RequestValidationError

log = logging.getLogger("krea2")

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
DEV_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]


class HostCheckMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Host header is not in the allow-list (DNS-rebinding guard) with 421."""

    def __init__(self, app, allowed_hosts: list[str]):
        super().__init__(app)
        self.allowed = {h.lower() for h in allowed_hosts}

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        hostname = host.rsplit(":", 1)[0] if host.count(":") == 1 or host.startswith("[") else host
        hostname = hostname.strip("[]").lower()
        if hostname not in self.allowed:
            return JSONResponse({"detail": f"untrusted Host header {host!r}"}, status_code=421)
        return await call_next(request)


class AppState:
    def __init__(self, config: AppConfig, engine: Engine):
        self.config = config
        self.engine = engine
        self.models = ModelRegistry(config)
        self.outputs = OutputStore(config.paths.outputs)
        self.bus = EventBus()
        self.queue = GenerationQueue(engine, self.outputs, self.models, self.bus)


def build_engine(config: AppConfig, kind: str | None = None) -> Engine:
    kind = kind or os.environ.get("KREA2_ENGINE") or config.engine
    if kind == "fake":
        from .engine.fake_engine import FakeEngine

        return FakeEngine()
    from .engine.native_engine import NativeEngine

    return NativeEngine(config)


def create_app(config: AppConfig | None = None, engine: Engine | None = None, engine_kind: str | None = None) -> FastAPI:
    config = config or load_config()
    engine = engine or build_engine(config, engine_kind)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.bus.bind_loop(asyncio.get_running_loop())
        if config.optimization.warmup:
            await asyncio.to_thread(_safe_warmup, engine)
        yield
        state.queue.close()
        engine.close()

    app = FastAPI(title="krea2-darask", version=__version__, lifespan=lifespan)
    state = AppState(config, engine)
    app.state.krea2 = state

    app.add_middleware(HostCheckMiddleware, allowed_hosts=config.server.allowed_hosts)
    if config.server.dev_cors:
        app.add_middleware(CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    def st() -> AppState:
        return state

    # -- meta --------------------------------------------------------------------------------------
    @app.get("/api/health")
    def health(s: AppState = Depends(st)) -> dict:
        return {"ok": True, "version": __version__, "engine": s.engine.name, "pending": s.queue.pending()}

    @app.get("/api/capabilities")
    def capabilities(s: AppState = Depends(st)) -> dict:
        caps = dict(s.engine.capabilities())
        caps["version"] = __version__
        caps["defaults"] = s.config.defaults.model_dump()
        caps["optimization"] = s.config.optimization.model_dump()
        return caps

    @app.get("/api/models")
    def models(s: AppState = Depends(st)) -> dict:
        return s.models.all_models().model_dump()

    @app.get("/api/presets")
    def presets() -> dict:
        return presets_payload()

    class TokenQuery(BaseModel):
        prompt: str = ""
        width: int = 1024
        height: int = 1024
        hires: bool = False

    @app.post("/api/tokens")
    def tokens(q: TokenQuery, s: AppState = Depends(st)) -> dict:
        text = s.engine.count_text_tokens(q.prompt)
        w, h = (q.width * 2, q.height * 2) if q.hires else (q.width, q.height)
        return {
            "text_tokens": text,
            "truncated": text >= 512,
            "total_tokens": total_tokens(w, h, text),
            "bsa_eligible": total_tokens(w, h, text) >= s.config.optimization.bsa_min_tokens,
            "hires_warning": hires_edge_warning(q.width, q.height) if q.hires else None,
        }

    # -- generation --------------------------------------------------------------------------------
    @app.post("/api/generate", status_code=202)
    def generate(req: GenerateRequest, s: AppState = Depends(st)) -> dict:
        try:
            job = s.queue.submit(req)
        except RequestValidationError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"job": job.to_dict(), "warnings": req.warnings()}

    @app.get("/api/jobs")
    def jobs(s: AppState = Depends(st)) -> dict:
        return {"jobs": s.queue.recent(), "pending": s.queue.pending()}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, s: AppState = Depends(st)) -> dict:
        j = s.queue.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        return j.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str, s: AppState = Depends(st)) -> dict:
        return {"cancelled": s.queue.cancel(job_id)}

    @app.websocket("/ws/progress")
    async def ws_progress(ws: WebSocket, s: AppState = Depends(st)) -> None:
        origin = ws.headers.get("origin")
        if origin and not _origin_allowed(origin, s.config):
            await ws.close(code=1008)
            return
        await ws.accept()
        q = s.bus.subscribe()
        try:
            await ws.send_json({"type": "hello", "pending": s.queue.pending(), "jobs": s.queue.recent(5)})
            while True:
                event = await q.get()
                await ws.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            s.bus.unsubscribe(q)

    # -- outputs -----------------------------------------------------------------------------------
    @app.get("/api/outputs")
    def outputs(limit: int = 200, offset: int = 0, s: AppState = Depends(st)) -> dict:
        limit = max(1, min(limit, 1000))
        return {"items": [e.to_dict() for e in s.outputs.list(limit, max(0, offset))]}

    def _oid(output_id: str) -> str:
        try:
            return validate_output_id(output_id)
        except UnsafePathError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/outputs/{day}/{name}/image")
    def output_image(day: str, name: str, s: AppState = Depends(st)) -> FileResponse:
        oid = _oid(f"{day}/{name}")
        try:
            return FileResponse(s.outputs.path_of(oid))
        except (FileNotFoundError, UnsafePathError) as exc:
            raise HTTPException(404, "not found") from exc

    @app.get("/api/outputs/{day}/{name}/thumb")
    def output_thumb(day: str, name: str, s: AppState = Depends(st)) -> FileResponse:
        oid = _oid(f"{day}/{name}")
        try:
            return FileResponse(s.outputs.thumbnail(oid), media_type="image/webp")
        except (FileNotFoundError, UnsafePathError) as exc:
            raise HTTPException(404, "not found") from exc

    @app.get("/api/outputs/{day}/{name}/metadata")
    def output_metadata(day: str, name: str, s: AppState = Depends(st)) -> dict:
        oid = _oid(f"{day}/{name}")
        try:
            return s.outputs.metadata(oid)
        except (FileNotFoundError, UnsafePathError) as exc:
            raise HTTPException(404, "not found") from exc

    @app.delete("/api/outputs/{day}/{name}")
    def output_delete(day: str, name: str, s: AppState = Depends(st)) -> dict:
        oid = _oid(f"{day}/{name}")
        try:
            s.outputs.delete(oid)
        except (FileNotFoundError, UnsafePathError) as exc:
            raise HTTPException(404, "not found") from exc
        return {"deleted": oid}

    @app.post("/api/open-folder")
    def open_folder(s: AppState = Depends(st)) -> dict:
        path = s.outputs.root
        if os.name == "nt":
            os.startfile(path)  # noqa: S606 - local desktop app, fixed path
            return {"opened": str(path)}
        return {"opened": None, "path": str(path)}

    # -- metadata restore (drag & drop) --------------------------------------------------------------
    @app.post("/api/metadata/parse")
    async def metadata_parse(file: UploadFile = File(...)) -> dict:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large")
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                result = read_image(img)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"not a readable image: {exc}") from exc
        request_dict: dict[str, Any] = result.request
        validated: dict | None = None
        errors: list[str] = list(result.warnings)
        if request_dict:
            try:
                validated = GenerateRequest.model_validate(request_dict).model_dump()
            except ValidationError as exc:
                errors.append(f"restored settings did not validate fully: {exc.error_count()} field(s)")
                validated = _lenient_request(request_dict)
        return {
            "source": result.source,
            "request": validated,
            "parameters": result.parameters,
            "krea2gui": result.krea2gui.model_dump(exclude_none=True) if result.krea2gui else None,
            "warnings": errors,
        }

    # -- static GUI (production, same-origin) -------------------------------------------------------
    dist = REPO_ROOT / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app


def _lenient_request(raw: dict) -> dict | None:
    """Drop fields one by one until the rest validates (best-effort restore from foreign metadata)."""
    data = dict(raw)
    for _ in range(len(data)):
        try:
            return GenerateRequest.model_validate(data).model_dump()
        except ValidationError as exc:
            bad = {e["loc"][0] for e in exc.errors() if e.get("loc")}
            if not bad:
                return None
            for key in bad:
                data.pop(key, None)
    return None


def _origin_allowed(origin: str, config: AppConfig) -> bool:
    allowed = {f"http://{h}:{config.server.port}" for h in config.server.allowed_hosts}
    if config.server.dev_cors:
        allowed.update(DEV_ORIGINS)
    return origin in allowed


def _safe_warmup(engine: Engine) -> None:
    try:
        engine.warmup()
    except Exception as exc:  # noqa: BLE001
        log.error("warmup failed (continuing eager/uncompiled): %s", exc)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="krea2-darask")
    parser.add_argument("--fake", action="store_true", help="use the deterministic fake engine (no GPU)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dev-cors", action="store_true", help="allow the Vite dev server origin")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    if args.port:
        config.server.port = args.port
    if args.dev_cors:
        config.server.dev_cors = True
    if args.no_warmup:
        config.optimization.warmup = False
    if config.server.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("server.host must stay loopback (127.0.0.1); this GUI is local-only")

    import uvicorn

    app = create_app(config, engine_kind="fake" if args.fake else None)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info", ws="websockets")


if __name__ == "__main__":
    main()
