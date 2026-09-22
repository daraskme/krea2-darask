from __future__ import annotations

import argparse
import asyncio
import json
import io
import secrets
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import queue
import subprocess
import webbrowser
from urllib.parse import urlparse

from aiohttp import web
from PIL import Image, UnidentifiedImageError

from . import __version__
from .config import PROJECT_ROOT, SettingsStore, load_config, output_root, output_url, safe_output_path
from .discovery import discover_loras, discover_models
from .engine import KreaEngine
from .jobs import JobManager


def error(code: str, message: str, status: int = 400, details=None):
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return web.json_response(body, status=status)


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except json.JSONDecodeError:
        return error("invalid_json", "Request body is not valid JSON")
    except ValueError as exc:
        return error("invalid_request", str(exc))
    except KeyError as exc:
        return error("not_found", f"Not found: {exc.args[0]}", 404)


@web.middleware
async def local_only(request, handler):
    host = request.host.rsplit(":", 1)[0].strip("[]").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return error("forbidden_host", "Only loopback Host headers are accepted", 403)
    origin = request.headers.get("Origin")
    if origin:
        parsed = urlparse(origin)
        expected_port = int(request.app["config"]["server"]["port"])
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port not in {None, expected_port}:
            return error("forbidden_origin", "Cross-origin requests are not accepted", 403)
    if request.path in {"/api/generate", "/api/upscale", "/api/load-model", "/api/load-loras", "/api/settings", "/api/open-storage-folder"} and request.method in {"POST", "PUT", "PATCH"}:
        if request.content_type != "application/json":
            return error("unsupported_media_type", "Content-Type must be application/json", 415)
    return await handler(request)


def create_app(config=None, engine=None) -> web.Application:
    config = config or load_config()
    engine = engine or KreaEngine(config)
    manager = JobManager(engine, int(config["server"]["max_pending_jobs"]))
    settings = SettingsStore()
    app = web.Application(middlewares=[errors, local_only], client_max_size=26 * 1024 * 1024)
    app["config"] = config
    app["jobs"] = manager
    app["settings"] = settings

    async def health(_):
        return web.json_response({"ok": True, "engine": "krea2-studio", "version": __version__})

    async def capabilities(_):
        models = discover_models(config)["items"]
        raw = any(x["available"] and x["family"] == "raw" for x in models)
        fast_id = discover_loras(config).get("fast4_lora_id")
        upscaler_path = Path(config["paths"]["upscaler"])
        spandrel_installed = importlib.util.find_spec("spandrel") is not None
        upscaler_available = spandrel_installed and upscaler_path.is_file()
        if not spandrel_installed:
            upscaler_reason = "Spandrel is unavailable."
        else:
            upscaler_reason = None if upscaler_available else f"Model was not found: {upscaler_path}"
        return web.json_response({
            "engine": "krea2-studio", "version": __version__,
            "device": device_info(),
            "features": {
                "hires": True, "metadata": ["png_itxt", "exif_user_comment", "json_sidecar"],
                "explicit_model_loading": True, "automatic_lora_loading": True,
                "attention_backends": ["sdpa", "sage2"],
                "upscaler": {
                    "backend": "spandrel" if upscaler_available else "pillow_lanczos",
                    "model": upscaler_path.name if upscaler_available else None,
                    "reason": upscaler_reason,
                },
                "presets": [
                    {"id": "turbo8", "available": True},
                    {"id": "fast4", "available": bool(fast_id), "reason": None if fast_id else "4-step LoRA not found"},
                    {"id": "raw", "available": raw, "reason": None if raw else "No supported raw model is registered"},
                ],
                "vc_attention": {"available": False, "reason": "No validated independent implementation is installed."},
            },
            "queue": {"max_pending": manager.max_pending},
        })

    async def state(_): return web.json_response(manager.state())
    async def models(_): return web.json_response(discover_models(config))
    async def loras(_): return web.json_response(discover_loras(config))
    async def get_settings(_): return web.json_response(settings.load())
    async def storage(_):
        model_root = Path(config["paths"]["model_root"]).resolve()
        lora_root = Path(config["paths"]["lora_root"]).resolve()
        return web.json_response({
            "model_root": str(model_root), "lora_root": str(lora_root),
            "models": [{"id": item["id"], "name": item["name"], "path": str(Path(item["path"]).resolve())}
                       for item in discover_models(config)["items"]],
        })
    async def open_storage_folder(request):
        kind = str((await request.json()).get("kind", ""))
        folders = {
            "models": Path(config["paths"]["model_root"]).resolve(),
            "loras": Path(config["paths"]["lora_root"]).resolve(),
        }
        if kind not in folders:
            raise ValueError("Unknown storage folder")
        folder = folders[kind]
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return web.json_response({"opened": True, "path": str(folder)})
    async def put_settings(request):
        data = {**settings.load(), **await request.json()}
        if data["output_dir"] != settings.load()["output_dir"]:
            status = manager.state()
            if status["active_job_id"] or status["queue_length"]:
                return error("busy", "Wait for active jobs before changing the output directory", 409)
        return web.json_response(settings.save(data))
    async def import_image(request):
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "image":
            return error("invalid_image", "Choose an image file")
        chunks = bytearray()
        while chunk := await part.read_chunk():
            chunks.extend(chunk)
            if len(chunks) > 25 * 1024 * 1024:
                return error("too_large", "Image file must be at most 25 MiB", 413)
        try:
            with Image.open(io.BytesIO(chunks)) as opened:
                if opened.width > 4096 or opened.height > 4096 or opened.width < 16 or opened.height < 16:
                    return error("invalid_size", "Source image must be between 16 and 4096 pixels per side")
                image = opened.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            return error("invalid_image", "The file is not a supported image")
        folder = output_root() / "imports" / datetime.now().strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"source_{secrets.token_hex(12)}.png"
        await asyncio.to_thread(image.save, path, "PNG")
        return web.json_response({"image_url": output_url(path), "width": image.width, "height": image.height, "filename": part.filename or "image"}, status=201)
    async def generate(request):
        try:
            result = manager.submit(await request.json())
        except queue.Full:
            return error("queue_full", "The generation queue is full", 429)
        return web.json_response(result, status=202)
    async def upscale_image(request):
        try:
            result = manager.submit(await request.json(), operation="upscale")
        except queue.Full:
            return error("queue_full", "The generation queue is full", 429)
        return web.json_response(result, status=202)
    async def load_model(request):
        try:
            result = manager.submit(await request.json(), operation="model_load")
        except queue.Full:
            return error("queue_full", "The engine queue is full", 429)
        return web.json_response(result, status=202)
    async def load_loras(request):
        try:
            result = manager.submit(await request.json(), operation="loras_load")
        except queue.Full:
            return error("queue_full", "The engine queue is full", 429)
        return web.json_response(result, status=202)
    async def get_job(request): return web.json_response(manager.get(request.match_info["job_id"]))
    async def cancel_job(request): return web.json_response(manager.cancel(request.match_info["job_id"]))
    async def history(request): return web.json_response({"items": manager.history(int(request.query.get("limit", 50)))})
    async def output(request):
        path = safe_output_path(request.match_info["path"])
        if not path.is_file():
            raise KeyError(request.match_info["path"])
        return web.FileResponse(path)
    async def open_output(_):
        root = output_root()
        root.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            subprocess.Popen(["xdg-open", str(root)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.startfile(root)  # type: ignore[attr-defined]
        return web.json_response({"opened": True})

    app.router.add_get("/health", health)
    app.router.add_get("/api/capabilities", capabilities)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/models", models)
    app.router.add_get("/api/loras", loras)
    app.router.add_get("/api/settings", get_settings)
    app.router.add_get("/api/storage", storage)
    app.router.add_post("/api/open-storage-folder", open_storage_folder)
    app.router.add_put("/api/settings", put_settings)
    app.router.add_post("/api/import-image", import_image)
    app.router.add_post("/api/generate", generate)
    app.router.add_post("/api/upscale", upscale_image)
    app.router.add_post("/api/load-model", load_model)
    app.router.add_post("/api/load-loras", load_loras)
    app.router.add_get("/api/jobs/{job_id}", get_job)
    app.router.add_post("/api/jobs/{job_id}/cancel", cancel_job)
    app.router.add_get("/api/history", history)
    app.router.add_post("/api/open-output-folder", open_output)
    app.router.add_get("/outputs/{path:.+}", output)
    web_root = PROJECT_ROOT / "web"
    if (web_root / "index.html").is_file():
        if (web_root / "assets").is_dir():
            app.router.add_static("/assets", web_root / "assets", show_index=False)
        for filename in ("app.js", "styles.css"):
            if (web_root / filename).is_file():
                app.router.add_get(f"/{filename}", lambda _, p=web_root / filename: web.FileResponse(p))
        app.router.add_get("/", lambda _: web.FileResponse(web_root / "index.html"))

    async def cleanup(_app):
        await asyncio.to_thread(manager.shutdown)
    app.on_cleanup.append(cleanup)
    return app


def device_info():
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=3, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, memory_mib = [part.strip() for part in completed.stdout.strip().split(",", 1)]
        return {"type": "cuda", "name": name, "vram_bytes": int(memory_mib) * 1024 * 1024, "capability": None}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {"type": "unavailable", "name": None, "vram_bytes": None, "capability": None}


def main():
    parser = argparse.ArgumentParser(description="Krea 2 Studio")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    url = f"http://{config['server']['host']}:{config['server']['port']}"
    if config["server"]["open_browser"] and not args.no_browser:
        asyncio.get_event_loop_policy()
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    web.run_app(create_app(config), host=config["server"]["host"], port=config["server"]["port"], print=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
