from __future__ import annotations

import copy
from datetime import datetime, timezone
import queue
import threading
import traceback
import uuid
from typing import Any

from .config import OUTPUT_ROOT
from .engine import GenerationCancelled, KreaEngine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, engine: KreaEngine, max_pending: int = 8):
        self.engine = engine
        self.max_pending = max_pending
        self.jobs: dict[str, dict[str, Any]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_pending)
        self._lock = threading.RLock()
        self._active_id: str | None = None
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="krea2-gpu-worker", daemon=True)
        self._thread.start()

    def submit(self, request: dict[str, Any], operation: str = "generate") -> dict[str, Any]:
        if operation not in {"generate", "upscale"}:
            raise ValueError(f"Unknown operation: {operation}")
        job_id = uuid.uuid4().hex
        with self._lock:
            finished = [key for key, value in self.jobs.items() if value["status"] in {"completed", "failed", "cancelled"}]
            for old_id in finished[:-500]:
                self.jobs.pop(old_id, None)
                self._cancel.pop(old_id, None)
            if self._queue.full():
                raise queue.Full()
            position = self._queue.qsize() + (1 if self._active_id else 0)
            self.jobs[job_id] = {
                "id": job_id, "status": "queued", "progress": 0.0, "stage": "queued", "message": "待機中",
                "created_at": _now(), "started_at": None, "completed_at": None,
                "request": copy.deepcopy(request), "error": None, "result": None, "operation": operation,
            }
            self._cancel[job_id] = threading.Event()
            self._queue.put_nowait(job_id)
        return {"job_id": job_id, "status": "queued", "position": position}

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self.jobs[job_id])

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] in {"completed", "failed", "cancelled"}:
                return {"id": job_id, "status": job["status"], "cancel_requested": False}
            self._cancel[job_id].set()
            if job["status"] == "queued":
                job.update(status="cancelled", stage="cancelled", message="キャンセル済み", completed_at=_now())
            return {"id": job_id, "status": job["status"], "cancel_requested": True}

    def state(self) -> dict[str, Any]:
        with self._lock:
            active = self.jobs.get(self._active_id) if self._active_id else None
            status = "idle"
            if active:
                status = "loading" if active["stage"] == "loading" else "generating"
            return {
                "status": status, "active_job_id": self._active_id, "queue_length": self._queue.qsize(),
                "model_id": self.engine.model_id, "attention_backend": self.engine.attention_backend,
                "last_error": next((x["error"] for x in reversed(list(self.jobs.values())) if x["error"]), None),
            }

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            completed = [x for x in reversed(list(self.jobs.values())) if x["status"] == "completed"]
            known_urls = {x.get("result", {}).get("image_url") for x in completed if x.get("result")}
        saved = []
        for sidecar in sorted(OUTPUT_ROOT.glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            png = sidecar.with_suffix(".png")
            if not png.is_file():
                continue
            image_url = "/outputs/" + png.relative_to(OUTPUT_ROOT).as_posix()
            if image_url in known_urls:
                continue
            try:
                import json
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            saved.append({
                "id": "saved-" + sidecar.stem, "status": "completed", "progress": 1.0,
                "stage": "completed", "message": "保存済み", "created_at": metadata.get("created_at"),
                "started_at": None, "completed_at": metadata.get("created_at"), "request": metadata,
                "error": None, "result": {
                    "image_url": image_url,
                    "metadata_url": "/outputs/" + sidecar.relative_to(OUTPUT_ROOT).as_posix(),
                    "width": metadata.get("final_width", metadata.get("width")),
                    "height": metadata.get("final_height", metadata.get("height")),
                    "seed": metadata.get("seed"), "elapsed_seconds": metadata.get("elapsed_seconds"),
                },
            })
        merged = completed + saved
        merged.sort(key=lambda x: x.get("completed_at") or "", reverse=True)
        return copy.deepcopy(merged[: max(1, min(int(limit), 200))])

    def shutdown(self) -> None:
        self._stopping.set()
        with self._lock:
            for event in self._cancel.values():
                event.set()
        self._thread.join(timeout=30)
        if not self._thread.is_alive():
            self.engine.close()

    def _update_progress(self, job_id: str, value: float, stage: str, message: str) -> None:
        with self._lock:
            status = "loading" if stage == "loading" else "running"
            self.jobs[job_id].update(
                status=status, progress=max(0.0, min(1.0, float(value))), stage=stage, message=message
            )

    def _worker(self) -> None:
        while not self._stopping.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job_id is None:
                return
            with self._lock:
                job = self.jobs[job_id]
                if job["status"] == "cancelled":
                    self._queue.task_done()
                    continue
                self._active_id = job_id
                job.update(status="loading", stage="loading", message="モデルを準備中", started_at=_now())
                request = copy.deepcopy(job["request"])
                cancel_event = self._cancel[job_id]
            try:
                execute = self.engine.upscale_existing if job.get("operation") == "upscale" else self.engine.generate
                result = execute(
                    request,
                    lambda value, stage, message: self._update_progress(job_id, value, stage, message),
                    cancel_event.is_set,
                )
                with self._lock:
                    self.jobs[job_id].update(
                        status="completed", progress=1.0, stage="completed", message="完了",
                        result=result, completed_at=_now(),
                    )
            except GenerationCancelled:
                with self._lock:
                    self.jobs[job_id].update(status="cancelled", stage="cancelled", message="キャンセル済み", completed_at=_now())
            except Exception as exc:
                with self._lock:
                    self.jobs[job_id].update(
                        status="failed", stage="failed", message="生成に失敗しました", completed_at=_now(),
                        error={"code": exc.__class__.__name__, "message": str(exc), "details": traceback.format_exc(limit=8)},
                    )
            finally:
                with self._lock:
                    self._active_id = None
                self._queue.task_done()
