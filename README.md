# Krea 2 Studio

Krea 2 Studio is a local Windows GUI and independent Diffusers generation engine for Krea 2. It never imports, starts, or sends requests to ComfyUI. Existing model files under `D:/comfyui-models` are read directly and remain in place.

## Start

1. Run `setup.bat` once. It creates this project's own `.venv` and installs the pinned CUDA 13.0 runtime, Diffusers, SageAttention 2, and the web API dependencies.
2. Run `start.bat`. The app opens at `http://127.0.0.1:8189`.
3. Enter a prompt, choose Turbo 8 or Turbo 4, optionally add ordered LoRAs, and generate.
4. To create a high-resolution version, select a completed image and run the separate high-resolution job.

The server binds only to loopback and disables model downloads. Edit `config.local.toml` (using `config.example.toml` as a template) when model paths differ. Additional models must explicitly declare `family = "turbo"` or `family = "raw"`; self-contained local Diffusers folders are supported.

## Local model support

- The installed Krea 2 Turbo BF16 single-file transformer, Qwen3-VL text encoder, and Qwen Image VAE are converted and loaded by the independent engine.
- The installed 4-step distillation LoRA and style LoRAs are converted to Diffusers adapters. Adapter order and strengths are recorded in every result.
- ComfyUI ConvRot INT8 checkpoints are listed as unavailable with the exact reason. Their quantization layout is not silently interpreted as BF16.
- The local UltraSharp model is used through Spandrel for high-resolution generation. Lanczos is used with a visible reason if the model cannot load.
- Raw generation remains disabled until a compatible BF16 or self-contained Diffusers raw model is registered.

Outputs are lossless PNG files under `outputs/YYYY-MM-DD`. Generation settings are stored in PNG iTXt, standard EXIF UserComment, and a JSON sidecar. The gallery is rebuilt from those sidecars after restart.

## Presets

- Turbo 8 uses the distilled Turbo model, 8 steps by default, and Krea guidance `0`.
- Turbo 4 automatically loads the local 4-step adapter at strength `1.0`, uses 4 steps, and Krea guidance `0`.
- High resolution is a separate job that takes an already saved project output. It upscales that image and runs a deterministic refinement schedule without repeating base generation. The 4-step adapter is removed while selected style adapters stay active.

SageAttention is scoped to the Krea model and preserves text padding masks through key/value compaction. Unsupported kernel cases fall back to PyTorch SDPA and the fallback count is written to metadata. VC Attention is not exposed because no validated independent implementation is installed.
