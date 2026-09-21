# Krea 2 Studio — implementation and acceptance plan

## Scope and architecture

- User decision: an independent generation engine. No ComfyUI server, imports, node execution or runtime dependency. Existing workflow JSON files are reference data only, never instructions to execute.
- Local Python engine using PyTorch and Hugging Face Diffusers Krea2, with a loopback-only aiohttp API and a polished Japanese browser GUI. Single GPU worker, bounded queue, progress, cancellation, error recovery and history.
- Hardware verified: RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB VRAM, sm_120, driver 597.06. User reports 128 GB RAM. Default BF16 with GPU residency and supported SageAttention; avoid CPU offload by default on this GPU. Release old model/adapters on switching; report actual active backend and memory.
- Read existing local model directories without copying weights. Support local Diffusers folders and existing Krea2 BF16 single-file components through an explicit key conversion loader. Quantized Comfy INT8/convrot is not a drop-in Diffusers format; reject unsupported formats accurately, rather than treating quantized bytes as BF16 or claiming an optimization not active.

## Features

1. Clean responsive single-page studio: prompt, aspect/size, preset, seed, model selector, ordered multiple LoRA controls (weight, enabled state, add/remove), advanced settings, image preview, progress, gallery and reuse settings.
2. Turbo 8-step baseline; Turbo 4-step requires the installed distillation LoRA at the correct strength and scheduler. Krea guidance scale is 0 for turbo (Comfy CFG 1 is not copied as Diffusers scale 1). Raw preset has CFG guidance and its own appropriate schedule. The filename alone must not silently imply model family.
3. Existing Comfy-format Krea2 LoRA conversion using supported Diffusers conversion, including alpha scaling, explicit incompatible-key failure and deterministic stacking/reset. Never combine 4-step distillation with incompatible acceleration.
4. Updated user requirement: generation and Hires upscale are separate screens and jobs. Normal T2I ends after saving. The Hires screen selects an existing output image, restores its prompt/model/style settings, upscales that actual source, and performs a correctly scheduled image-to-image refinement pass without rerunning base T2I. Keep the 4-step distillation adapter out of the 8-step refinement pass, retain selected style adapters, and record the source plus actual refinement settings. Prefer the installed UltraSharp model through Spandrel; explicitly label any interpolation fallback.
5. Optimization: inference mode/BF16; correctly implemented per-model Sage2 attention with GQA/shape/scale/mask handling and SDPA fallback; bounded prompt embedding cache with correct keys; VAE tiling at high resolution; optional compile only when supported and measured. Do not blindly transplant sparse Comfy nodes or low-step cache-skipping.
6. VC Attention evaluated against arXiv:2609.15810v1. Existing local unofficial node is not the paper's algorithm and has correctness issues. No VC claims or enabled toggle without a valid independent implementation and numerical/image validation. Present precise availability reason in optimization details/documentation.
7. Outputs always under this project's `outputs/` directory (date subfolders are fine), accessible through UI and an open-folder action. Save prompt, negative prompt, actual resolved seed, model/components, ordered enabled LoRAs and strengths, sampling schedule, dimensions, actual backend, software version and all refinement settings in real EXIF UserComment plus PNG iTXt; JSON sidecar as redundancy. Unicode round trip, lossless PNG by default. Settings reuse from app outputs.
8. Windows one-click launcher, requirements/version constraints, local config/model discovery, concise usage and limitations. No silent model downloads. Existing weights should suffice; small public architecture/tokenizer assets may be obtained when necessary with clear attribution and local caching.

## Review and verification responsibilities

- Astra: source research, plan review, adversarial implementation review, verify acceptance evidence. Sol: all product code implementation and fixes. Primary agent: coordination, research, review documents and runtime/UI verification.
- Tests: settings/path validation, ordered LoRA lifecycle and unsupported formats, Unicode EXIF/iTXt round trip, seed handling, bounded queue/cancellation/error, metadata completeness, real attention numerical checks for GQA/masks/odd lengths.
- Integration: actual local GPU generation with benign prompt at modest resolution, actual 4-step adapter + two style LoRAs, model switching, hires pass, saved-image readback, repeated warm run timing and backend report. Separately report anything not executable; no simulated image presented as generation.
- GUI: inspect real browser at desktop and narrow viewport, run generation through UI, inspect errors, gallery and settings reuse.
- Document cold vs warm measurements and limitations; do not reuse benchmark claims from workflow notes as new measurements.

## Astra adversarial plan review — required corrections

1. Diffusers Krea2 uses padding masks. Sage2 cannot simply replace attention while ignoring the mask. Preserve exact masking via supported attention or validated key compaction; report fallback counts and actual backend.
2. Hires is an explicit img2img loop: VAE mean/std normalization, latent packing, shifted sigma schedule, strength-dependent start, correct image/noise interpolation and scheduler begin index. Turbo remains distilled during the 8-step refinement pass.
3. Existing LoRAs include lora_down/up names. Normalize to A/B and fold alpha/rank correctly before the upstream converter. Failed loads must roll back all adapters.
4. Only Turbo BF16 is currently available in a directly adaptable unquantized format. Raw must remain unavailable until a supported model is configured. Do not report Raw GPU verification without an executable checkpoint.
5. Component conversion requires exact key and shape matching, preserved precision-sensitive tensors and no remaining meta tensors. No silent partial state loading.
6. A project-specific Python environment and versioned dependency files are required; Comfy's environment may be used for read-only comparison, never required by the shipped launcher.
7. Verify base → multiple LoRAs → base restoration, actual distillation adapter use/removal, Unicode EXIF, cancellation recovery and backend behavior.

Review status: architecture approved subject to these corrections. Product implementation assigned to Sol engine and Sol GUI agents. Astra will review the resulting code and integration evidence.

2026-09-22 user refinement: generation and Hires must be separate. Earlier combined-pass GPU tests validate the loading, attention, adapters and refinement math, but the separate source-image flow requires fresh API/UI/GPU acceptance before completion.

## Completion evidence — 2026-09-22

Sol implemented the independent engine and GUI; Astra approved the reviewed implementation after adversarial fixes and independent output readback. See `docs/ASTRA_REVIEW.md` for findings, evidence and limits.

- Separate Hires API: an existing 1024 image became 2048, with zero base-generation passes, one refinement pass, both style LoRAs retained, and matching EXIF/iTXt/sidecar metadata. The original image's SHA256 stayed unchanged.
- Separate Hires GUI: browser submission of a 256 image produced 512 after the 2048 job, confirming the small-image path after VAE tiling. Evidence: `bench/split-ui-acceptance.json`.
- Final browser checks covered desktop 1440px and narrow 390px, history-to-Hires metadata restoration, distinct tabs, no horizontal overflow, and generation-settings reuse with actual seed 424242, fast4 steps fixed to 4, guidance 0, and two style adapters.
- Final verification: 17 unit tests passed, JavaScript syntax check passed, and Git whitespace check passed. Queue-full shutdown and model-release/attention-reapplication regressions are included.
- VC Attention, Comfy ConvRot INT8, bias-delta LoRA and Raw Hires remain explicitly unavailable. Only one compatible checkpoint is installed, so different-checkpoint GPU switching is not claimed as tested.

## Follow-up requirements — local collection and explicit loading

- Physically copy Krea2 weights and supporting tokenizer assets to private project `models/`, preserve original files and existing LoRA IDs, and switch `config.local.toml` only after SHA256 verification. Completed: 65 files / 75.04 GiB; 43 LoRAs discovered, 42 supported. The additional 18 style variants passed Astra's CPU/meta shape verification. All copies and local configuration remain Git-ignored.
- Add an actual model-load button with separate selected/loaded model status. Apply selected LoRAs automatically after a model is loaded; before loading, retain selection and apply it with the explicit load. Use the single worker queue, separate control jobs from image history, and prevent stale selection requests from replacing newer requests.
- Canvas presets: portrait 512×2048, 576×1728, 768×1344, 832×1216, 896×1152; landscape reverses each pair; square defaults to 1024×1024. Keep manual dimensions and show approximate ratio labels where appropriate.
- The user is updating the GPU driver. The real app server was stopped and its CUDA process released. Follow-up verification uses CPU unit tests and a separate fake-engine browser fixture; no new GPU inference or automatic restart is permitted during this work.
- Follow-up acceptance completed: 19 CPU unit tests, JavaScript syntax check, all 11 canvas presets clicked and dimensions read back, 528×960 custom input validation, 390px layout, explicit preload/automatic LoRA requests, no startup or model-selection auto-load, truthful failure state, and browser-reload revision recovery. `bench/ui-followup-acceptance.json` records the CPU-only browser evidence. Actual GPU acceptance for the newly added preload controls remains deferred until the driver update is finished.
