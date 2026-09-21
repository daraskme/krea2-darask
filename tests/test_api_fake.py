from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image

from app.engine.fake_engine import FakeEngine, render_fake_image
from app.metadata import read_image
from app.schemas import GenerateRequest


def test_fake_engine_deterministic():
    a = render_fake_image(42, 256, 384, "hello")
    b = render_fake_image(42, 256, 384, "hello")
    c = render_fake_image(43, 256, 384, "hello")
    d = render_fake_image(42, 256, 384, "other")
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()
    assert a.tobytes() != d.tobytes()
    assert a.size == (256, 384)


def test_fake_engine_generate_batch():
    eng = FakeEngine(step_delay_s=0.0)
    steps_seen = []
    outs = eng.generate(GenerateRequest(prompt="p", width=256, height=256, batch_size=2, steps=4), 7,
                        lambda s, n, info: steps_seen.append((s, n)))
    assert [o.seed for o in outs] == [7, 8]
    assert steps_seen[-1] == (4, 4) and len(steps_seen) == 8
    assert eng.capabilities()["fake"] is True


def _wait_job(client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_models_and_presets_endpoints(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert [m["name"] for m in body["diffusion_models"]] == ["krea2_turbo_bf16.safetensors"]
    assert len(body["loras"]) == 2
    assert body["hf_text_encoders"][-1] == "Qwen3-VL-4B-Instruct"
    r = client.get("/api/presets")
    assert r.status_code == 200
    assert {p["id"] for p in r.json()["workflows"]} >= {"fast_4step", "turbo_8step", "hires_2x"}
    assert any(p["id"] == "832x2048" for p in r.json()["resolutions"])
    caps = client.get("/api/capabilities").json()
    assert caps["engine"] == "fake" and "defaults" in caps


def test_host_header_check(client):
    r = client.get("/api/health", headers={"Host": "evil.example.com"})
    assert r.status_code == 421
    assert client.get("/api/health", headers={"Host": "localhost:8765"}).status_code == 200
    assert client.get("/api/health", headers={"Host": "127.0.0.1"}).status_code == 200


def test_generate_validation_errors(client):
    r = client.post("/api/generate", json={"prompt": "x", "lora_4step": True, "bsa": True})
    assert r.status_code == 400
    assert "4-step LoRA" in r.json()["detail"]
    r = client.post("/api/generate", json={"prompt": "x", "width": "abc"})
    assert r.status_code == 422


def test_generate_flow_outputs_metadata_and_gallery(client, app_config):
    req = {
        "prompt": "a red fox, in snow",
        "negative_prompt": "blurry",
        "preset": "fast_4step",
        "width": 256,
        "height": 384,
        "steps": 4,
        "seed": 123,
        "transformer": "krea2_turbo_bf16.safetensors",
        "loras": [{"name": "style_a.safetensors", "strength": 0.6}],
    }
    with client.websocket_connect("/ws/progress") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        r = client.post("/api/generate", json=req)
        assert r.status_code == 202, r.text
        job_id = r.json()["job"]["id"]
        types = []
        for _ in range(200):
            ev = ws.receive_json()
            types.append(ev["type"])
            if ev["type"] == "finished" and ev["job"]["id"] == job_id:
                break
    assert "progress" in types and "image" in types and types[-1] == "finished"
    job = _wait_job(client, job_id)
    assert job["status"] == "done", job
    assert len(job["outputs"]) == 1
    output_id = job["outputs"][0]

    # file on disk with both chunks
    path: Path = app_config.paths.outputs / output_id
    assert path.is_file()
    with Image.open(path) as im:
        assert im.size == (256, 384)
        assert "parameters" in im.text and "krea2gui" in im.text
    parsed = read_image(path)
    assert parsed.source == "krea2gui"
    assert parsed.krea2gui.request.prompt == req["prompt"]
    assert parsed.krea2gui.result.seed == 123
    assert parsed.krea2gui.result.lora_hashes["style_a.safetensors"]  # hashed by the model registry
    assert parsed.krea2gui.result.transformer_hash

    # gallery + per-image endpoints
    items = client.get("/api/outputs").json()["items"]
    assert items[0]["id"] == output_id
    day, name = output_id.split("/")
    assert client.get(f"/api/outputs/{day}/{name}/image").status_code == 200
    thumb = client.get(f"/api/outputs/{day}/{name}/thumb")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/webp"
    meta = client.get(f"/api/outputs/{day}/{name}/metadata").json()
    assert meta["request"]["seed"] == 123 and meta["request"]["loras"][0]["strength"] == 0.6

    # drag & drop restore
    files = {"file": ("x.png", path.read_bytes(), "image/png")}
    r = client.post("/api/metadata/parse", files=files)
    assert r.status_code == 200
    restored = r.json()
    assert restored["source"] == "krea2gui"
    assert restored["request"]["prompt"] == req["prompt"]
    assert restored["request"]["width"] == 256 and restored["request"]["seed"] == 123

    # traversal attempts on the outputs API
    assert client.get("/api/outputs/2026-01-01/..%2F..%2Fsecret.png/image").status_code in (400, 404)
    assert client.get(f"/api/outputs/{day}/{name.replace('.png', '.exe')}/image").status_code == 400

    # delete
    assert client.delete(f"/api/outputs/{day}/{name}").status_code == 200
    assert not path.exists()


def test_metadata_parse_rejects_garbage(client):
    r = client.post("/api/metadata/parse", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    r = client.post("/api/metadata/parse", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["source"] == "none" and r.json()["request"] is None


def test_hires_request_via_api(client):
    r = client.post("/api/generate", json={"prompt": "x", "preset": "hires_2x", "width": 128, "height": 128, "seed": 1})
    assert r.status_code == 202
    job = _wait_job(client, r.json()["job"]["id"])
    assert job["status"] == "done"
    assert job["results"][0]["width"] == 256 and job["results"][0]["height"] == 256
