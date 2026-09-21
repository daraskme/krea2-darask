from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import copy
import secrets
from pathlib import Path
import time
from typing import Any

from .attention import Sage2KreaAttnProcessor
from .config import OUTPUT_ROOT, safe_output_path
from .discovery import discover_loras, discover_models, get_model_spec, resolve_lora
from .loaders import PromptEmbeddingCache, load_lora_file, load_text_encoder, load_transformer, load_vae
from .metadata import read_image_metadata, save_image
from .upscale import clear_upscaler_cache, upscale


class GenerationCancelled(Exception):
    pass


class KreaEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.pipe = None
        self.model_id: str | None = None
        self.attention_backend: str | None = None
        self.attention_reason: str | None = None
        self._attention_processor = None
        self.active_loras: list[dict[str, Any]] = []
        self.loaded_preset: str | None = None
        self.selection_revision: int | None = None
        self.cache = PromptEmbeddingCache(config["engine"]["embedding_cache_size"])

    def close(self) -> None:
        self.pipe = None
        self.model_id = None
        self.active_loras.clear()
        self.loaded_preset = None
        self.selection_revision = None
        self.cache.clear()
        clear_upscaler_cache()
        self._attention_processor = None
        self.attention_backend = None
        self.attention_reason = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_model(self, model_id: str) -> None:
        if self.model_id == model_id and self.pipe is not None:
            return
        spec = get_model_spec(self.config, model_id)
        item = next(x for x in discover_models(self.config)["items"] if x["id"] == model_id)
        if not item["available"]:
            raise ValueError(item.get("reason", "Selected model is unavailable"))
        self.close()
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for Krea 2 generation")
        if spec["kind"] == "diffusers":
            self.pipe = Krea2Pipeline.from_pretrained(
                spec["path"], torch_dtype=torch.bfloat16, local_files_only=True,
            ).to("cuda")
        else:
            paths = self.config["paths"]
            transformer = load_transformer(Path(spec["path"]), "cuda")
            text_encoder, tokenizer = load_text_encoder(
                Path(paths["text_encoder"]), Path(paths["qwen_config"]), "cuda"
            )
            vae = load_vae(Path(paths["vae"]), "cuda")
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=True,
                base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400,
            )
            self.pipe = Krea2Pipeline(
                scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                transformer=transformer, is_distilled=bool(spec["distilled"]),
            )
        self.model_id = model_id
        self.active_loras = []

    def _set_attention(self, requested: str) -> None:
        import torch
        from diffusers.models.transformers.transformer_krea2 import Krea2AttnProcessor

        resolved = requested
        if requested == "auto":
            resolved = "sage2" if torch.cuda.is_available() else "sdpa"
        if resolved == self.attention_backend and self._attention_processor is not None:
            return
        try:
            processor = Sage2KreaAttnProcessor() if resolved == "sage2" else Krea2AttnProcessor()
            self.attention_reason = None
        except (ImportError, RuntimeError) as exc:
            if requested != "auto":
                raise
            resolved = "sdpa"
            processor = Krea2AttnProcessor()
            self.attention_reason = f"SageAttention unavailable; SDPA selected: {exc}"
        self.pipe.transformer.set_attn_processor(processor)
        self._attention_processor = processor
        self.attention_backend = resolved

    def _configure_loras(self, requested: list[dict[str, Any]], preset: str) -> list[dict[str, Any]]:
        selected = [dict(x) for x in requested if x.get("enabled", True) and float(x.get("weight", 1.0)) != 0]
        catalog = discover_loras(self.config)
        by_id = {x["id"]: x for x in catalog["items"]}
        for entry in selected:
            if entry.get("id") not in by_id:
                raise ValueError(f"Unknown LoRA: {entry.get('id')}")
            if not by_id[entry["id"]]["available"]:
                raise ValueError(by_id[entry["id"]].get("reason", f"LoRA is unavailable: {entry['id']}"))
            entry["weight"] = float(entry.get("weight", 1.0))
            if not -4.0 <= entry["weight"] <= 4.0:
                raise ValueError("LoRA weight must be between -4 and 4")
        selected_distillers = [x for x in selected if by_id[x["id"]]["category"] == "distillation"]
        if preset == "fast4":
            fast_id = catalog.get("fast4_lora_id")
            if not fast_id:
                raise ValueError("The required 4-step distillation LoRA was not found")
            if selected_distillers and any(x["id"] != fast_id for x in selected_distillers):
                raise ValueError("fast4 cannot be combined with another distillation adapter")
            selected = [x for x in selected if x["id"] != fast_id]
            selected.insert(0, {"id": fast_id, "weight": 1.0, "enabled": True, "role": "fast4"})
        elif selected_distillers:
            raise ValueError("Distillation adapters are only accepted by the fast4 preset")

        if selected == self.active_loras:
            return copy.deepcopy(selected)
        previous = copy.deepcopy(self.active_loras)
        prepared = [(entry, load_lora_file(resolve_lora(self.config, entry["id"]))) for entry in selected]

        def install(entries_and_states):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                self.active_loras = []
                raise
            self.active_loras = []
            names: list[str] = []
            weights: list[float] = []
            for index, (entry, state) in enumerate(entries_and_states):
                name = f"adapter_{index}"
                self.pipe.load_lora_weights(state, adapter_name=name)
                if name not in getattr(self.pipe.transformer, "peft_config", {}):
                    raise ValueError(f"LoRA {entry['id']} did not register compatible transformer layers")
                names.append(name)
                weights.append(entry["weight"])
            if names:
                self.pipe.set_adapters(names, adapter_weights=weights)

        try:
            install(prepared)
        except Exception as original:
            try:
                rollback = [(entry, load_lora_file(resolve_lora(self.config, entry["id"]))) for entry in previous]
                install(rollback)
                self.active_loras = previous
            except Exception as rollback_error:
                # The pipeline may now contain a partially installed adapter.
                # Drop it completely so an empty active_loras list can never
                # be mistaken for a known-clean loaded pipeline.
                self.close()
                raise RuntimeError(f"LoRA update failed and rollback failed: {rollback_error}") from original
            raise
        self.active_loras = copy.deepcopy(selected)
        return copy.deepcopy(selected)

    def _resolve_control_request(self, request: dict[str, Any]) -> dict[str, Any]:
        model_id = str(request.get("model_id", "")).strip()
        if not model_id:
            raise ValueError("model_id is required")
        spec = get_model_spec(self.config, model_id)
        model = next((item for item in discover_models(self.config)["items"] if item["id"] == model_id), None)
        if model is None or not model["available"]:
            raise ValueError((model or {}).get("reason", f"Model is unavailable: {model_id}"))
        preset = str(request.get("preset", "turbo8"))
        if preset not in {"turbo8", "fast4", "raw"}:
            raise ValueError(f"Unknown preset: {preset}")
        expected_family = "raw" if preset == "raw" else "turbo"
        if spec["family"] != expected_family:
            raise ValueError(f"The {preset} preset requires a registered {expected_family} model")
        backend = str(request.get("attention_backend", self.config["engine"]["attention_backend"]))
        if backend not in {"auto", "sdpa", "sage2"}:
            raise ValueError(f"Unknown attention backend: {backend}")
        loras = request.get("loras", [])
        if not isinstance(loras, list) or any(not isinstance(item, dict) for item in loras):
            raise ValueError("loras must be an array of objects")
        revision = request.get("selection_revision")
        if revision is not None:
            revision = int(revision)
            if revision < 0:
                raise ValueError("selection_revision must be non-negative")
        return {
            "model_id": model_id, "preset": preset, "attention_backend": backend,
            "loras": copy.deepcopy(loras), "selection_revision": revision,
        }

    def preload(
        self, request: dict[str, Any], progress: Callable[[float, str, str], None], cancelled: Callable[[], bool]
    ) -> dict[str, Any]:
        resolved = self._resolve_control_request(request)
        started = time.perf_counter()
        progress(0.05, "loading", "モデルを読み込み中")
        self._load_model(resolved["model_id"])
        if cancelled():
            raise GenerationCancelled()
        self._set_attention(resolved["attention_backend"])
        progress(0.75, "configuring", "LoRAを反映中")
        active = self._configure_loras(resolved["loras"], resolved["preset"])
        self.loaded_preset = resolved["preset"]
        self.selection_revision = resolved["selection_revision"]
        return {
            "loaded": True, "model_id": self.model_id, "preset": self.loaded_preset,
            "attention_backend": self.attention_backend, "loras": active,
            "selection_revision": self.selection_revision, "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def load_loras(
        self, request: dict[str, Any], progress: Callable[[float, str, str], None], cancelled: Callable[[], bool]
    ) -> dict[str, Any]:
        resolved = self._resolve_control_request(request)
        if self.pipe is None or self.model_id != resolved["model_id"]:
            raise ValueError("Load the selected model before applying LoRAs")
        started = time.perf_counter()
        if cancelled():
            raise GenerationCancelled()
        progress(0.2, "configuring", "LoRAを読み込み中")
        self._set_attention(resolved["attention_backend"])
        active = self._configure_loras(resolved["loras"], resolved["preset"])
        self.loaded_preset = resolved["preset"]
        self.selection_revision = resolved["selection_revision"]
        return {
            "loaded": True, "model_id": self.model_id, "preset": self.loaded_preset,
            "attention_backend": self.attention_backend, "loras": active,
            "selection_revision": self.selection_revision, "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _embeds(self, prompt: str, max_length: int = 512):
        import torch

        key = (self.model_id, prompt, max_length)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        with torch.inference_mode():
            value = self.pipe.encode_prompt(prompt=prompt, device="cuda", max_sequence_length=max_length)
        value = tuple(tensor.detach() for tensor in value)
        self.cache.put(key, value)
        return value

    def generate(
        self,
        request: dict[str, Any],
        progress: Callable[[float, str, str], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, Any]:
        import torch

        total_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        resolved = self._resolve_request(request)
        model_was_loaded = self.pipe is not None and self.model_id == resolved["model_id"]
        progress(0.01, "loading", "モデルを準備中")
        self._load_model(resolved["model_id"])
        self._set_attention(resolved["attention_backend"])
        if hasattr(self._attention_processor, "fallback_count"):
            self._attention_processor.fallback_count = 0
        active_loras = self._configure_loras(resolved["loras"], resolved["preset"])
        self.loaded_preset = resolved["preset"]
        # A generation request owns an immutable adapter snapshot that may differ
        # from the latest UI selection. Keep the actual adapters visible, but do
        # not claim that a UI selection revision is still applied.
        self.selection_revision = None
        if max(resolved["width"], resolved["height"]) >= int(self.config["engine"]["vae_tiling_threshold"]):
            self.pipe.vae.enable_tiling()
        else:
            self.pipe.vae.disable_tiling()
        if cancelled():
            raise GenerationCancelled()

        generator = torch.Generator(device="cuda").manual_seed(resolved["seed"])
        prompt_embeds, prompt_mask = self._embeds(resolved["prompt"])
        negative_embeds = negative_mask = None
        if resolved["guidance_scale"] > 0:
            negative_embeds, negative_mask = self._embeds(resolved["negative_prompt"])

        def callback(_pipe, step, _timestep, kwargs):
            if cancelled():
                raise GenerationCancelled()
            fraction = (step + 1) / resolved["steps"]
            progress(0.08 + fraction * 0.72, "base", f"生成 {step + 1}/{resolved['steps']}")
            return kwargs

        started = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(
                prompt=None, negative_prompt=None, prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask,
                negative_prompt_embeds=negative_embeds, negative_prompt_embeds_mask=negative_mask,
                width=resolved["width"], height=resolved["height"], num_inference_steps=resolved["steps"],
                guidance_scale=resolved["guidance_scale"], generator=generator,
                callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
            )
        image = result.images[0]
        base_sigmas = [float(x) for x in self.pipe.scheduler.sigmas.detach().cpu().tolist()]
        pass_records = [{
            "kind": "base", "steps": resolved["steps"], "guidance_scale": resolved["guidance_scale"],
            "sigmas": base_sigmas, "scheduler": self.pipe.scheduler.__class__.__name__,
            "scheduler_config": dict(self.pipe.scheduler.config),
            "loras": [{"id": x["id"], "weight": x["weight"]} for x in active_loras],
        }]
        if cancelled():
            raise GenerationCancelled()
        elapsed = time.perf_counter() - started
        if cancelled():
            raise GenerationCancelled()
        performance = {
            "inference_seconds": round(elapsed, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
            "model_was_loaded": model_was_loaded,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        metadata = self._metadata(resolved, active_loras, pass_records, elapsed, performance)
        progress(0.97, "saving", "メタデータを保存中")
        stem = f"krea2_{datetime.now().strftime('%H%M%S_%f')}_{resolved['seed']}_{secrets.token_hex(3)}"
        png_path, json_path = save_image(image, OUTPUT_ROOT, stem, metadata)
        relative_png = png_path.relative_to(OUTPUT_ROOT).as_posix()
        relative_json = json_path.relative_to(OUTPUT_ROOT).as_posix()
        return {
            "image_url": f"/outputs/{relative_png}", "metadata_url": f"/outputs/{relative_json}",
            "width": image.width, "height": image.height, "seed": resolved["seed"], "elapsed_seconds": round(elapsed, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
        }

    def upscale_existing(
        self,
        request: dict[str, Any],
        progress: Callable[[float, str, str], None],
        cancelled: Callable[[], bool],
    ) -> dict[str, Any]:
        """Upscale and refine an existing project output without running a base text-to-image pass."""
        import torch
        from PIL import Image

        total_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        resolved, source_path, parent = self._resolve_upscale_request(request)
        model_was_loaded = self.pipe is not None and self.model_id == resolved["model_id"]
        progress(0.01, "loading", "モデルを準備中")
        self._load_model(resolved["model_id"])
        self._set_attention(resolved["attention_backend"])
        if hasattr(self._attention_processor, "fallback_count"):
            self._attention_processor.fallback_count = 0
        active_loras = self._configure_loras(resolved["loras"], "turbo8")
        self.loaded_preset = "turbo8"
        self.selection_revision = None
        if cancelled():
            raise GenerationCancelled()
        prompt_embeds, prompt_mask = self._embeds(resolved["prompt"])
        if cancelled():
            raise GenerationCancelled()
        with Image.open(source_path) as opened:
            source_image = opened.convert("RGB").copy()
        started = time.perf_counter()
        progress(0.08, "hires_upscale", "高解像度化中")
        image, refine_record = self._refine(
            source_image, resolved, prompt_embeds, prompt_mask, cancelled, progress
        )
        inference_seconds = time.perf_counter() - started
        if cancelled():
            raise GenerationCancelled()
        source_relative = source_path.relative_to(OUTPUT_ROOT).as_posix()
        passes = [{
            "kind": "hires_refine", "steps": resolved["hires"]["refine_steps"],
            "denoise_strength": resolved["hires"]["denoise_strength"], "distillation_adapter_removed": True,
            "loras": [{"id": x["id"], "weight": x["weight"]} for x in active_loras], **refine_record,
        }]
        performance = {
            "inference_seconds": round(inference_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
            "model_was_loaded": model_was_loaded,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        metadata = self._metadata(resolved, active_loras, passes, inference_seconds, performance)
        metadata["operation"] = "upscale"
        metadata["source"] = {
            "image_url": f"/outputs/{source_relative}", "relative_path": source_relative,
            "metadata_url": f"/outputs/{source_path.with_suffix('.json').relative_to(OUTPUT_ROOT).as_posix()}",
            "schema": parent.get("schema"), "created_at": parent.get("created_at"), "seed": parent.get("seed"),
        }
        progress(0.97, "saving", "高解像度画像を保存中")
        stem = f"krea2_hires_{datetime.now().strftime('%H%M%S_%f')}_{resolved['seed']}_{secrets.token_hex(3)}"
        png_path, json_path = save_image(image, OUTPUT_ROOT, stem, metadata)
        return {
            "image_url": f"/outputs/{png_path.relative_to(OUTPUT_ROOT).as_posix()}",
            "metadata_url": f"/outputs/{json_path.relative_to(OUTPUT_ROOT).as_posix()}",
            "width": image.width, "height": image.height, "seed": resolved["seed"],
            "elapsed_seconds": round(inference_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
            "source_image_url": f"/outputs/{source_relative}",
        }

    def _resolve_upscale_request(self, request: dict[str, Any]):
        source = str(request.get("source_image", "")).strip()
        if not source:
            raise ValueError("source_image is required for high-resolution refinement")
        relative = source[len("/outputs/"):] if source.startswith("/outputs/") else source
        source_path = safe_output_path(relative)
        if source_path.suffix.lower() != ".png" or not source_path.is_file():
            raise ValueError("source_image must name an existing PNG under this project's outputs directory")
        try:
            parent = read_image_metadata(source_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read source image metadata: {exc}") from exc
        parent_model = parent.get("model") if isinstance(parent.get("model"), dict) else {}
        model_id = str(request.get("model_id") or parent_model.get("id") or self.config["engine"]["default_model"])
        spec = get_model_spec(self.config, model_id)
        if spec["family"] != "turbo":
            raise ValueError("High-resolution refinement currently requires a registered turbo model")
        prompt = str(request.get("prompt") if request.get("prompt") is not None else parent.get("prompt", "")).strip()
        if not prompt or len(prompt) > 4000:
            raise ValueError("prompt is required and must be at most 4000 characters")
        seed_value = request.get("seed", parent.get("seed"))
        seed = secrets.randbits(32) if seed_value in (None, "", -1) else int(seed_value)
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2^32-1")
        backend = str(request.get("attention_backend") or parent.get("attention_backend") or self.config["engine"]["attention_backend"])
        if backend not in {"auto", "sdpa", "sage2"}:
            raise ValueError(f"Unknown attention backend: {backend}")
        requested_loras = list(request["loras"] if "loras" in request else parent.get("loras", []))
        catalog = {x["id"]: x for x in discover_loras(self.config)["items"]}
        unknown_loras = [entry.get("id") for entry in requested_loras if entry.get("id") not in catalog]
        if unknown_loras:
            raise ValueError(f"Source settings reference unknown LoRA(s): {unknown_loras}")
        style_loras = [
            {**entry, "enabled": entry.get("enabled", True)} for entry in requested_loras
            if entry.get("id") in catalog and catalog[entry["id"]]["category"] != "distillation"
        ]
        hires = {
            "enabled": True, "scale": float(request.get("scale", 1.5)),
            "method": str(request.get("method", "auto")), "refine_steps": int(request.get("refine_steps", 8)),
            "denoise_strength": float(request.get("denoise_strength", 0.3)),
        }
        if not 1.0 <= hires["scale"] <= 2.0 or not 1 <= hires["refine_steps"] <= 50 or not 0.05 <= hires["denoise_strength"] <= 0.95:
            raise ValueError("Invalid high-resolution settings")
        if hires["method"] not in {"auto", "neural", "lanczos"}:
            raise ValueError(f"Unknown high-resolution method: {hires['method']}")
        from PIL import Image
        with Image.open(source_path) as source_image:
            width, height = source_image.size
        final_width = int(round(width * hires["scale"] / 16) * 16)
        final_height = int(round(height * hires["scale"] / 16) * 16)
        if final_width > 4096 or final_height > 4096 or final_width * final_height > 4096 * 4096:
            raise ValueError(f"High-resolution output is too large: {final_width}x{final_height}; maximum is 4096x4096")
        resolved = {
            "prompt": prompt, "negative_prompt": str(request.get("negative_prompt", parent.get("negative_prompt", ""))),
            "model_id": model_id, "preset": "hires", "width": width, "height": height, "seed": seed,
            "steps": hires["refine_steps"], "guidance_scale": 0.0, "loras": style_loras,
            "attention_backend": backend, "hires": hires,
        }
        return resolved, source_path, parent

    def _refine(self, image, request, prompt_embeds, prompt_mask, cancelled, progress):
        import numpy as np
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler

        hires = request["hires"]
        width = int(round(image.width * hires["scale"] / 16) * 16)
        height = int(round(image.height * hires["scale"] / 16) * 16)
        if width > 4096 or height > 4096 or width * height > 4096 * 4096:
            raise ValueError(f"High-resolution output is too large: {width}x{height}; maximum is 4096x4096")
        if max(width, height) >= int(self.config["engine"]["vae_tiling_threshold"]):
            self.pipe.vae.enable_tiling()
        else:
            self.pipe.vae.disable_tiling()
        upscaler_path = None if hires["method"] == "lanczos" else Path(self.config["paths"]["upscaler"])
        image, upscale_info = upscale(image, scale=hires["scale"], model_path=upscaler_path, device="cuda")
        if image.size != (width, height):
            from PIL import Image as PILImage
            image = image.resize((width, height), resample=PILImage.Resampling.LANCZOS)
        pixels = self.pipe.image_processor.preprocess(image).to("cuda", dtype=self.pipe.vae.dtype).unsqueeze(2)
        with torch.inference_mode():
            source = self.pipe.vae.encode(pixels).latent_dist.mode()[:, :, 0]
        mean = torch.tensor(self.pipe.vae.config.latents_mean, device="cuda", dtype=source.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.pipe.vae.config.latents_std, device="cuda", dtype=source.dtype).view(1, -1, 1, 1)
        source = (source - mean) / std
        packed = self.pipe._pack_latents(source, 1, source.shape[1], source.shape[2], source.shape[3])
        steps = hires["refine_steps"]
        sigmas = np.linspace(hires["denoise_strength"], hires["denoise_strength"] / steps, steps).tolist()
        probe = FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        probe.set_timesteps(sigmas=sigmas, device="cuda", mu=1.15)
        sigma = probe.sigmas[0].to(packed.dtype)
        noise = torch.randn(packed.shape, generator=torch.Generator(device="cuda").manual_seed(request["seed"] + 1), device="cuda", dtype=packed.dtype)
        start = (1 - sigma) * packed + sigma * noise

        def callback(_pipe, step, _timestep, kwargs):
            if cancelled():
                raise GenerationCancelled()
            progress(0.12 + 0.83 * (step + 1) / steps, "hires_refine", f"仕上げ {step + 1}/{steps}")
            return kwargs

        with torch.inference_mode():
            result = self.pipe(
                prompt=None, prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask,
                width=width, height=height, num_inference_steps=steps, sigmas=sigmas, guidance_scale=0.0,
                latents=start, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
            )
        return result.images[0], {
            "sigmas": [float(x) for x in probe.sigmas.cpu().tolist()], "width": width, "height": height,
            "upscaler": upscale_info,
        }

    def _resolve_request(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = str(request.get("prompt", "")).strip()
        if not prompt or len(prompt) > 4000:
            raise ValueError("prompt is required and must be at most 4000 characters")
        model_id = str(request.get("model_id") or self.config["engine"]["default_model"])
        spec = get_model_spec(self.config, model_id)
        preset = str(request.get("preset", "turbo8"))
        if preset not in {"turbo8", "fast4", "raw"}:
            raise ValueError(f"Unknown preset: {preset}")
        if preset == "raw" and spec["family"] != "raw":
            raise ValueError("The raw preset requires a registered raw model")
        if preset in {"turbo8", "fast4"} and spec["family"] != "turbo":
            raise ValueError("Turbo presets require a registered turbo model")
        width, height = int(request.get("width", 1024)), int(request.get("height", 1024))
        if any(x < 256 or x > 4096 or x % 16 for x in (width, height)):
            raise ValueError("width and height must be multiples of 16 between 256 and 4096")
        seed = request.get("seed")
        seed = secrets.randbits(32) if seed in (None, "", -1) else int(seed)
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2^32-1")
        if preset == "fast4":
            steps, guidance = 4, 0.0
        elif preset == "turbo8":
            steps, guidance = int(request.get("steps") or 8), 0.0
        else:
            steps, guidance = int(request.get("steps") or 28), float(request.get("guidance_scale", 4.5))
        if not 1 <= steps <= 100 or not 0 <= guidance <= 20:
            raise ValueError("Invalid steps or guidance scale")
        hires = dict(request.get("hires") or {})
        hires = {
            "enabled": bool(hires.get("enabled", False)), "scale": float(hires.get("scale", 1.5)),
            "method": str(hires.get("method", "auto")), "refine_steps": int(hires.get("refine_steps", 8)),
            "denoise_strength": float(hires.get("denoise_strength", 0.3)),
        }
        if not 1 <= hires["refine_steps"] <= 50 or not 1.0 <= hires["scale"] <= 2.0 or not 0.05 <= hires["denoise_strength"] <= 0.95:
            raise ValueError("Invalid high-resolution settings")
        if hires["method"] not in {"auto", "neural", "lanczos"}:
            raise ValueError(f"Unknown high-resolution method: {hires['method']}")
        if hires["enabled"] and preset == "raw":
            raise ValueError("High-resolution refinement for raw models is not available in this version")
        if hires["enabled"]:
            raise ValueError("Generation and high-resolution refinement are separate jobs; use POST /api/upscale")
        backend = str(request.get("attention_backend", self.config["engine"]["attention_backend"]))
        if backend not in {"auto", "sdpa", "sage2"}:
            raise ValueError(f"Unknown attention backend: {backend}")
        return {
            "prompt": prompt, "negative_prompt": str(request.get("negative_prompt", "")), "model_id": model_id,
            "preset": preset, "width": width, "height": height, "seed": seed, "steps": steps,
            "guidance_scale": guidance, "loras": list(request.get("loras") or []),
            "attention_backend": backend,
            "hires": hires,
        }

    def _metadata(self, request, loras, passes, elapsed, performance):
        import torch
        spec = get_model_spec(self.config, request["model_id"])
        components = {"pipeline": spec["path"]} if spec["kind"] == "diffusers" else {
            "transformer": spec["path"], "text_encoder": self.config["paths"]["text_encoder"],
            "vae": self.config["paths"]["vae"],
        }
        return {
            "schema": "krea2-studio/v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": request["prompt"], "negative_prompt": request["negative_prompt"], "seed": request["seed"],
            "model": {"id": request["model_id"], "family": get_model_spec(self.config, request["model_id"])["family"]},
            "loras": [{"id": x["id"], "weight": x["weight"]} for x in loras],
            "preset": request["preset"], "width": request["width"], "height": request["height"],
            "steps": request["steps"], "guidance_scale": request["guidance_scale"], "passes": passes,
            "hires": request["hires"],
            "final_width": passes[-1].get("width", request["width"]),
            "final_height": passes[-1].get("height", request["height"]),
            "attention_backend": self.attention_backend,
            "attention_reason": self.attention_reason,
            "attention_fallback_count": getattr(self._attention_processor, "fallback_count", 0),
            "elapsed_seconds": round(elapsed, 3),
            "performance": performance,
            "software": {
                "krea2_studio": importlib.metadata.version("krea2-studio"), "torch": torch.__version__,
                "diffusers": importlib.metadata.version("diffusers"), "transformers": importlib.metadata.version("transformers"),
            },
            "components": components,
        }
