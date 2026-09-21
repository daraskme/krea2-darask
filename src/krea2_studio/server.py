from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import queue
import webbrowser
from urllib.parse import urlparse

from aiohttp import web

from . import __version__
from .config import OUTPUT_ROOT, PROJECT_ROOT, SettingsStore, load_config, safe_output_path
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
    if request.path in {"/api/generate", "/api/upscale", "/api/settings"} and request.method in {"POST", "PUT", "PATCH"}:
        if request.content_type != "application/json":
            return error("unsupported_media_type", "Content-Type must be application/json", 415)
    return await handler(request)


def create_app(config=None, engine=None) -> web.Application:
    config = config or load_config()
    engine = engine or KreaEngine(config)
    manager = JobManager(engine, int(config["server"]["max_pending_jobs"]))
    settings = SettingsStore()
    app = web.Application(middlewares=[errors, local_only], client_max_size=1024 * 1024)
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
        try:
            import spandrel  # noqa: F401
            upscaler_available = upscaler_path.is_file()
            upscaler_reason = None if upscaler_available else f"Model was not found: {upscaler_path}"
        except ImportError as exc:
            upscaler_available, upscaler_reason = False, f"Spandrel is unavailable: {exc}"
        return web.json_response({
            "engine": "krea2-studio", "version": __version__,
            "device": device_info(),
            "features": {
                "hires": True, "metadata": ["png_itxt", "exif_user_comment", "json_sidecar"],
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
    async def put_settings(request): return web.json_response(settings.save(await request.json()))
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
    async def get_job(request): return web.json_response(manager.get(request.match_info["job_id"]))
    async def cancel_job(request): return web.json_response(manager.cancel(request.match_info["job_id"]))
    async def history(request): return web.json_response({"items": manager.history(int(request.query.get("limit", 50)))})
    async def output(request):
        path = safe_output_path(request.match_info["path"])
        if not path.is_file():
            raise KeyError(request.match_info["path"])
        return web.FileResponse(path)
    async def open_output(_):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            return error("unsupported", "Open folder is currently supported on Windows", 501)
        os.startfile(OUTPUT_ROOT)  # type: ignore[attr-defined]
        return web.json_response({"opened": True})

    app.router.add_get("/health", health)
    app.router.add_get("/api/capabilities", capabilities)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/models", models)
    app.router.add_get("/api/loras", loras)
    app.router.add_get("/api/settings", get_settings)
    app.router.add_put("/api/settings", put_settings)
    app.router.add_post("/api/generate", generate)
    app.router.add_post("/api/upscale", upscale_image)
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
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"type": "cuda", "name": props.name, "vram_bytes": props.total_memory, "capability": list(torch.cuda.get_device_capability(0))}
    except Exception:
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
