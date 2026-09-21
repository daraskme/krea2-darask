from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from PIL import Image, ExifTags

from krea2_studio.config import OUTPUT_ROOT, safe_output_path, validate_settings
from krea2_studio.jobs import JobManager
from krea2_studio.loaders import PromptEmbeddingCache, normalize_krea_lora_state, transformer_source_key
from krea2_studio.metadata import EXIF_USER_COMMENT, decode_user_comment, read_image_metadata, save_image


class ConfigTests(unittest.TestCase):
    def test_dimensions_and_path_escape(self):
        self.assertEqual(validate_settings({"width": 512, "height": 768})["height"], 768)
        with self.assertRaises(ValueError):
            validate_settings({"width": 513})
        with self.assertRaises(ValueError):
            safe_output_path("../private.txt")


class MetadataTests(unittest.TestCase):
    def test_png_itxt_exif_ifd_and_sidecar_unicode(self):
        metadata = {"prompt": "日本語 🐈 café", "seed": 42, "loras": [{"id": "絵柄", "weight": 0.75}]}
        with tempfile.TemporaryDirectory() as tmp:
            png, sidecar = save_image(Image.new("RGB", (8, 8), "navy"), Path(tmp), "sample", metadata)
            self.assertEqual(read_image_metadata(png), metadata)
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), metadata)
            with Image.open(png) as loaded:
                comment = loaded.getexif().get_ifd(ExifTags.IFD.Exif)[EXIF_USER_COMMENT]
                self.assertEqual(json.loads(decode_user_comment(comment)), metadata)


class LoaderTests(unittest.TestCase):
    def test_transformer_mapping_special_cases(self):
        self.assertEqual(transformer_source_key("transformer_blocks.2.scale_shift_table"), "blocks.2.mod.lin")
        self.assertEqual(transformer_source_key("transformer_blocks.4.attn.norm_q.weight"), "blocks.4.attn.qknorm.qnorm.scale")
        self.assertEqual(transformer_source_key("text_fusion.refiner_blocks.1.ff.down.weight"), "txtfusion.refiner_blocks.1.mlp.down.weight")

    def test_lora_alpha_is_folded_into_b(self):
        import torch
        state = {
            "diffusion_model.blocks.0.attn.wq.lora_down.weight": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            "diffusion_model.blocks.0.attn.wq.lora_up.weight": torch.tensor([[6.0, 7.0], [8.0, 9.0]]),
            "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(1.0),
        }
        result = normalize_krea_lora_state(state)
        self.assertTrue(torch.equal(result["diffusion_model.blocks.0.attn.wq.lora_B.weight"], state["diffusion_model.blocks.0.attn.wq.lora_up.weight"] * 0.5))
        self.assertFalse(any(key.endswith("alpha") for key in result))

    def test_flattened_kohya_krea_lora_is_expanded(self):
        import torch
        state = {
            "lora_unet_txtfusion_refiner_blocks_1_attn_wo.lora_down.weight": torch.ones(2, 3),
            "lora_unet_txtfusion_refiner_blocks_1_attn_wo.lora_up.weight": torch.ones(3, 2),
        }
        result = normalize_krea_lora_state(state)
        self.assertIn("diffusion_model.txtfusion.refiner_blocks.1.attn.wo.lora_A.weight", result)

    def test_cache_is_lru_bounded(self):
        cache = PromptEmbeddingCache(2)
        cache.put(("a",), (1, 1)); cache.put(("b",), (2, 2)); cache.get(("a",)); cache.put(("c",), (3, 3))
        self.assertIsNone(cache.get(("b",)))
        self.assertEqual(len(cache), 2)


class _FakeEngine:
    model_id = None
    attention_backend = "sdpa"
    def __init__(self): self.closed = False
    def generate(self, request, progress, cancelled):
        for index in range(30):
            if cancelled():
                from krea2_studio.engine import GenerationCancelled
                raise GenerationCancelled()
            progress(index / 30, "base", "test")
            time.sleep(0.005)
        return {"image_url": "/outputs/x.png", "metadata_url": "/outputs/x.json", "seed": 1}
    def upscale_existing(self, request, progress, cancelled):
        progress(0.5, "hires_refine", "test")
        return {"image_url": "/outputs/upscaled.png", "metadata_url": "/outputs/upscaled.json", "seed": 1}
    def close(self): self.closed = True


class JobTests(unittest.TestCase):
    def test_cancel_active_and_worker_recovers(self):
        engine = _FakeEngine()
        jobs = JobManager(engine, 3)
        first = jobs.submit({})["job_id"]
        deadline = time.time() + 2
        while jobs.get(first)["status"] == "queued" and time.time() < deadline:
            time.sleep(0.01)
        jobs.cancel(first)
        second = jobs.submit({})["job_id"]
        while jobs.get(second)["status"] not in {"completed", "failed"} and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(jobs.get(first)["status"], "cancelled")
        self.assertEqual(jobs.get(second)["status"], "completed")
        jobs.shutdown()
        self.assertTrue(engine.closed)

    def test_shutdown_does_not_block_when_pending_queue_is_full(self):
        engine = _FakeEngine()
        jobs = JobManager(engine, 1)
        active = jobs.submit({})["job_id"]
        deadline = time.time() + 2
        while jobs.get(active)["status"] == "queued" and time.time() < deadline:
            time.sleep(0.01)
        jobs.submit({})
        started = time.perf_counter()
        jobs.shutdown()
        self.assertLess(time.perf_counter() - started, 2.0)
        self.assertTrue(engine.closed)

    def test_upscale_dispatch_never_calls_generate(self):
        engine = _FakeEngine()
        engine.generate = lambda *args: self.fail("base generation must not run for an upscale job")
        jobs = JobManager(engine, 2)
        job_id = jobs.submit({"source_image": "/outputs/source.png"}, operation="upscale")["job_id"]
        deadline = time.time() + 2
        while jobs.get(job_id)["status"] not in {"completed", "failed"} and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(jobs.get(job_id)["status"], "completed")
        self.assertEqual(jobs.get(job_id)["result"]["image_url"], "/outputs/upscaled.png")
        jobs.shutdown()


class SeparateHiresTests(unittest.TestCase):
    def test_generate_rejects_legacy_combined_hires(self):
        from krea2_studio.config import DEFAULT_CONFIG
        from krea2_studio.engine import KreaEngine
        engine = KreaEngine(DEFAULT_CONFIG)
        with self.assertRaisesRegex(ValueError, "separate jobs"):
            engine._resolve_request({"prompt": "test", "hires": {"enabled": True}})

    def test_upscale_requires_safe_existing_source(self):
        from krea2_studio.config import DEFAULT_CONFIG
        from krea2_studio.engine import KreaEngine
        engine = KreaEngine(DEFAULT_CONFIG)
        with self.assertRaisesRegex(ValueError, "source_image is required"):
            engine._resolve_upscale_request({})
        with self.assertRaises(ValueError):
            engine._resolve_upscale_request({"source_image": "../outside.png"})


class ModelLifecycleTests(unittest.TestCase):
    def test_attention_processor_is_reapplied_after_model_release(self):
        from krea2_studio.config import DEFAULT_CONFIG
        from krea2_studio.engine import KreaEngine

        class Transformer:
            def __init__(self): self.calls = 0
            def set_attn_processor(self, _processor): self.calls += 1
        class Pipeline:
            def __init__(self): self.transformer = Transformer()

        engine = KreaEngine(DEFAULT_CONFIG)
        first = Pipeline()
        engine.pipe = first
        engine.model_id = "first"
        engine._set_attention("sdpa")
        engine.cache.put(("old-model",), (1, 1))
        engine.active_loras = [{"id": "old"}]
        engine.close()
        second = Pipeline()
        engine.pipe = second
        engine.model_id = "second"
        engine._set_attention("sdpa")
        self.assertEqual(first.transformer.calls, 1)
        self.assertEqual(second.transformer.calls, 1)
        self.assertEqual(len(engine.cache), 0)
        self.assertEqual(engine.active_loras, [])
        self.assertEqual(engine.attention_backend, "sdpa")


class AttentionTests(unittest.TestCase):
    def test_mask_compacts_keys_but_retains_queries_and_gqa(self):
        import torch
        import torch.nn.functional as F
        from krea2_studio.attention import Sage2KreaAttnProcessor

        class FakeAttn:
            num_heads, num_kv_heads, head_dim = 4, 2, 2
            def __init__(self):
                torch.manual_seed(5)
                self.to_q = torch.nn.Linear(8, 8, bias=False)
                self.to_k = torch.nn.Linear(8, 4, bias=False)
                self.to_v = torch.nn.Linear(8, 4, bias=False)
                self.to_gate = torch.nn.Linear(8, 8, bias=False)
                self.to_out = [torch.nn.Identity()]
                self.norm_q = self.norm_k = torch.nn.Identity()

        processor = Sage2KreaAttnProcessor.__new__(Sage2KreaAttnProcessor)
        processor.fallback_count = 0
        processor._sageattn = lambda q, k, v, **_: F.scaled_dot_product_attention(q, k, v)
        attn, hidden = FakeAttn(), torch.randn(2, 13, 8)
        mask = torch.tensor([[1] * 9 + [0] * 4, [1, 0] * 6 + [1]], dtype=torch.bool)[:, None, None, :]
        actual = processor(attn, hidden, attention_mask=mask)
        q = attn.to_q(hidden).unflatten(-1, (4, 2))
        k = attn.to_k(hidden).unflatten(-1, (2, 2)).repeat_interleave(2, dim=2)
        v = attn.to_v(hidden).unflatten(-1, (2, 2)).repeat_interleave(2, dim=2)
        expected_rows = []
        for batch in range(2):
            valid = mask[batch, 0, 0]
            expected_rows.append(F.scaled_dot_product_attention(
                q[batch:batch + 1].transpose(1, 2), k[batch:batch + 1, valid].transpose(1, 2),
                v[batch:batch + 1, valid].transpose(1, 2),
            ).transpose(1, 2))
        expected = torch.cat(expected_rows).flatten(2, 3) * torch.sigmoid(attn.to_gate(hidden))
        self.assertEqual(actual.shape[1], hidden.shape[1])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
