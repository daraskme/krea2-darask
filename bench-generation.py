"""Measure warm Krea 2 image generation on the noxos GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from krea2_studio.config import OUTPUT_ROOT, load_config
from krea2_studio.discovery import discover_models
from krea2_studio.engine import KreaEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("turbo8", "fast4"), default="turbo8")
    parser.add_argument("--backend", choices=("sdpa", "sage2"), default="sdpa")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vae-tiling-threshold", type=int)
    parser.add_argument("--compile", action="store_true", help="Compile the loaded transformer with torch.compile")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--prompt", default="A red fox walking through a snowy forest at sunrise, detailed fur, natural light")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    config = load_config()
    if args.vae_tiling_threshold is not None:
        config["engine"]["vae_tiling_threshold"] = args.vae_tiling_threshold
    model_id = config["engine"]["default_model"]
    model = next((x for x in discover_models(config)["items"] if x["id"] == model_id), None)
    if model is None or not model["available"]:
        raise SystemExit(f"Model unavailable: {(model or {}).get('reason', model_id)}")
    request = {"prompt": args.prompt, "model_id": model_id, "preset": args.preset,
               "attention_backend": args.backend, "width": args.width,
               "height": args.height, "seed": args.seed}
    if args.preset == "turbo8":
        request["steps"] = args.steps
    engine = KreaEngine(config)
    records = []
    try:
        if args.compile:
            import torch
            engine.preload({"model_id": model_id, "preset": args.preset,
                            "attention_backend": args.backend, "loras": []},
                           lambda *_: None, lambda: False)
            engine.pipe.transformer = torch.compile(engine.pipe.transformer, mode="reduce-overhead")
        for index in range(args.repeats + 1):
            result = engine.generate(request, lambda *_: None, lambda: False)
            sidecar = OUTPUT_ROOT / result["metadata_url"].removeprefix("/outputs/")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            perf = metadata["performance"]
            if index:
                records.append({"image": result["image_url"], "inference_seconds": perf["inference_seconds"],
                                "total_seconds": perf["total_seconds"],
                                "peak_allocated_gib": perf["peak_allocated_gib"],
                                "fallback_count": metadata["attention_fallback_count"]})
            print(json.dumps({"run": index, "warmup": index == 0, "result": result,
                              "performance": perf}), flush=True)
    finally:
        engine.close()
    summary = {"preset": args.preset, "steps": args.steps if args.preset == "turbo8" else 4,
               "backend": args.backend, "compile": args.compile,
               "vae_tiling_threshold": config["engine"]["vae_tiling_threshold"],
               "width": args.width,
               "height": args.height, "seed": args.seed, "prompt": args.prompt,
               "median_inference_seconds": statistics.median(x["inference_seconds"] for x in records),
               "median_total_seconds": statistics.median(x["total_seconds"] for x in records),
               "runs": records}
    compiled = "compiled" if args.compile else "eager"
    target = Path("bench") / f"{args.preset}-{summary['steps']}step-{args.backend}-{args.width}x{args.height}-tile{config['engine']['vae_tiling_threshold']}-{compiled}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(target), "median_inference_seconds": summary["median_inference_seconds"]}))


if __name__ == "__main__":
    main()
