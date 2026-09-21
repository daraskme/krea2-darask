"""Single-worker generation queue with progress events for the WebSocket."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .engine.base import Engine, resolve_seed
from .models import ModelRegistry
from .outputs import OutputStore
from .schemas import GenerateRequest, GenerateResult, Krea2GuiMetadata

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    request: GenerateRequest
    seed: int
    status: str = "queued"  # queued | running | done | error | cancelled
    outputs: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    step: int = 0
    steps: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "seed": self.seed,
            "outputs": self.outputs,
            "results": self.results,
            "error": self.error,
            "step": self.step,
            "steps": self.steps,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "preset": self.request.preset,
            "hires": self.request.hires.enabled,
        }


class EventBus:
    """Fan-out of JSON events to asyncio subscribers (WebSocket connections), fed from the worker thread."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.history: deque[dict] = deque(maxlen=50)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        self.history.append(event)
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _push() -> None:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        try:
            loop.call_soon_threadsafe(_push)
        except RuntimeError:
            pass


class GenerationQueue:
    def __init__(self, engine: Engine, outputs: OutputStore, models: ModelRegistry, bus: EventBus | None = None):
        self.engine = engine
        self.outputs = outputs
        self.models = models
        self.bus = bus or EventBus()
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._cv = threading.Condition()
        self._stop = False
        self._worker = threading.Thread(target=self._run, name="krea2-worker", daemon=True)
        self._worker.start()

    # -- API --------------------------------------------------------------------------------------
    def submit(self, req: GenerateRequest) -> Job:
        req.validate()
        job = Job(id=uuid.uuid4().hex[:12], request=req, seed=resolve_seed(req.seed))
        with self._cv:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._cv.notify()
        self.bus.publish({"type": "queued", "job": job.to_dict(), "pending": self.pending()})
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def pending(self) -> int:
        with self._cv:
            return len(self._order)

    def cancel(self, job_id: str) -> bool:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return False
            job.status = "cancelled"
            try:
                self._order.remove(job_id)
            except ValueError:
                pass
        self.bus.publish({"type": "cancelled", "job": job.to_dict()})
        return True

    def recent(self, limit: int = 20) -> list[dict]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._worker.join(timeout=2)

    # -- worker --------------------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._order and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job = self._jobs[self._order.popleft()]
            self._execute(job)

    def _execute(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        self.bus.publish({"type": "started", "job": job.to_dict()})
        last_emit = 0.0

        def progress(step: int, steps: int, info: dict[str, Any]) -> None:
            nonlocal last_emit
            job.step, job.steps = step, steps
            now = time.time()
            if step == steps or now - last_emit >= 0.05:
                last_emit = now
                elapsed = now - (job.started or now)
                self.bus.publish(
                    {
                        "type": "progress",
                        "job_id": job.id,
                        "step": step,
                        "steps": steps,
                        "elapsed_s": round(elapsed, 3),
                        "it_s": round(step / elapsed, 3) if elapsed > 0 else None,
                        **info,
                    }
                )

        try:
            req = job.request
            for out in self.engine.generate(req, job.seed, progress):
                output_id = self.outputs.new_id(out.seed, req.output_format)
                lora_hashes = dict(out.lora_hashes)
                for lora in req.active_loras():
                    digest = lora.hash or self.models.hash_of("loras", lora.name)
                    if digest:
                        lora_hashes[lora.name] = digest
                result = GenerateResult(
                    output_id=output_id,
                    seed=out.seed,
                    width=out.width,
                    height=out.height,
                    elapsed_s=round(out.elapsed_s, 3),
                    steps=out.steps,
                    engine=self.engine.name,
                    attention_backend=out.attention_backend,
                    quant=out.quant,
                    bsa_used=out.bsa_used,
                    nag_used=out.nag_used,
                    compiled=out.compiled,
                    gpu_name=out.gpu_name,
                    text_tokens=out.text_tokens,
                    transformer_hash=out.transformer_hash or self.models.hash_of("diffusion_models", req.transformer),
                    lora_hashes=lora_hashes,
                )
                meta = Krea2GuiMetadata(request=req, result=result)
                self.outputs.save(out.image, meta, fmt=req.output_format)
                job.outputs.append(output_id)
                job.results.append(result.model_dump(exclude_none=True))
                self.bus.publish({"type": "image", "job_id": job.id, "output": output_id, "result": job.results[-1]})
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - report any engine failure to the client
            log.error("job %s failed: %s\n%s", job.id, exc, traceback.format_exc())
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.time()
            self.bus.publish({"type": "finished", "job": job.to_dict(), "pending": self.pending()})
