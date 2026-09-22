# Krea 2 Studio — noxos (NixOS)

This checkout is configured for the local noxos workstation: Intel i9-12900KS,
128 GiB RAM, NVIDIA RTX PRO 6000 Blackwell 96 GiB (sm_120), NVIDIA driver
595.99.02. The NixOS CUDA shell is `/etc/nixos#cuda`. Model paths default to
this checkout's private `models/` directory. Other machines are not a target.

## このPCでの推奨設定

通常は **Turbo 4＋4-step LoRA** を使用します。1024×1024 の生成部分は実測中央値
2.865秒で、Turbo 8 の6.646秒から短縮できました。2048×2048までは VAE
タイリングを無効にし、それを超えるサイズで有効にします。元のTurbo 8の画質を
優先したい場合はUIで切り替えられます。測定条件と比較結果は下表に記載しています。

## NixOS setup

```sh
git clone https://github.com/daraskme/krea2-darask.git
cd krea2-darask
nix shell nixpkgs#uv nixpkgs#python312 -c bash setup-nixos.sh
.venv/bin/hf auth login # after accepting the official model license in your browser
bash download-official-turbo.sh
bash download-4step-lora.sh # third-party distillation adapter for fast default
./run-nixos.sh
```

For an existing single-file model instead, place `krea2_turbo_bf16.safetensors`,
`qwen3vl_4b_bf16.safetensors`, `qwen_image_vae.safetensors`, and the
Qwen3-VL-4B-Instruct tokenizer/config folder at the paths in
`config.example.toml`. Put the 4-step LoRA under `models/loras/krea2/` if using
Turbo 4. Use `config.local.toml` for paths that differ. The
[official Turbo model](https://huggingface.co/krea/Krea-2-Turbo)
requires Hugging Face account access and acceptance of its license. The
download script pins a model revision and retrieves only the Diffusers
components, skipping the duplicate `turbo.safetensors` and sample images. It
creates a private `config.local.toml` if one does not exist. The CUDA shell
supplies `sm_120` compiler settings and the NVIDIA library path.
`requirements-nixos.txt` excludes the Windows-only Triton and SageAttention
wheels. The measured PyTorch SDPA backend is selected by default.

## モデルと LoRA の置き場所

画面の「モデルとスタイル」にモデルと LoRA の実際の保存フォルダーを表示し、
「開く」からファイルマネージャーで開けます。このPCの既定は次の通りです。

- 公式 Diffusers モデル: `models/krea2-turbo-diffusers/`
- Turbo 4 用 LoRA と追加 LoRA: `models/loras/krea2/` の `.safetensors`
- 単一ファイル形式のモデル構成: `models/diffusion_models/krea2/`、`models/text_encoders/`、`models/vae/`

上記はリポジトリ直下からの相対パスです。現在登録済みのモデルの実際のパスは
`config.local.toml` の `[[models]]` を確認してください。モデルや LoRA を追加したら
画面の「再読込」を押します。モデルファイルは Git に含まれません。

## 保存先と Hires への外部画像入力

生成画像、Hires の出力、読み込んだ外部画像は、既定では
`/run/media/hiroshi/ボリューム/生成物` に保存します。設定画面の「出力先フォルダー」には
存在する書き込み可能なフォルダーの絶対パスを入力し、「保存」で切り替えられます。
生成・アップスケール中は切り替えられません。以前の保存先と従来の `outputs/` の
画像も履歴から参照できます。外付けボリュームがマウントされていない場合は、
保存前にマウントしてください。

Hires タブでは履歴の画像に加え、PNG/JPEG/WebP をドラッグアンドドロップするか
「画像ファイルを選択」で読み込めます。外部画像には再描画プロンプトを入力してください。
読み込んだ画像は保存先の `imports/` に PNG として置かれます。ファイルは最大
25 MiB、辺の長さは 16〜4096 px です。仕上がりは最大 4096×4096 px です。

## Measured generation speed on noxos

The official [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) Diffusers
model runs in BF16 on the RTX PRO 6000 Blackwell. The community
[4-step distillation LoRA](https://huggingface.co/lvladikov/Krea2-Turbo-Distill-4step-LoRA)
is a third-party adapter. The app now opens with Turbo 4 selected; choose
Turbo 8 when its original output is preferred. The model stays resident in
GPU memory after **モデルを読み込む**. Repeating a prompt also reuses its text
embedding from the four-entry cache.

Measurements used PyTorch 2.14.0+cu130, NVIDIA driver 595.99.02, the same
fox-in-snow prompt and seed 12345. Times below are the pipeline call after
one warm-up, with the model already loaded. Text encoding, loading and PNG
saving add time to a full job. The 1024-pixel rows are medians of three runs;
1536-pixel rows are medians of two; 2048-pixel rows have one timed run.

| Size | Method | VAE tiles | Pipeline time | Peak PyTorch allocation |
| --- | --- | --- | ---: | ---: |
| 1024² | Turbo 8 | off | 6.646 s | 36.665 GiB |
| 1024² | Turbo 4 without LoRA | off | 2.327 s | 36.665 GiB |
| 1024² | Turbo 4 with LoRA | off | 2.865 s | 37.082 GiB |
| 1536² | Turbo 4 with LoRA | on | 8.168 s | 34.601 GiB |
| 1536² | Turbo 4 with LoRA | off | 8.018 s | 42.247 GiB |
| 2048² | Turbo 4 with LoRA | on | 25.602 s | 35.812 GiB |
| 2048² | Turbo 4 with LoRA | off | 16.395 s | 49.469 GiB |

At 1024², the LoRA option was 2.32× faster than Turbo 8. The unassisted
4-step image was faster still, but the LoRA image had clearer fur and snow
detail in this one visual comparison. At 2048², disabling VAE tiling saved
9.207 seconds while remaining well below 96 GiB VRAM. The default tiling
threshold is therefore **2049 px**: 2048 px and below use a full VAE decode;
larger images use tiles. At 1536² the timing difference was small.

`torch.compile(mode="reduce-overhead")` was also tried with Turbo 4 and the
LoRA at 1024². A warm generation took 2.398 s (two-run median), versus
2.865 s eager, but the first compiled generation took 70.0 s. In a second
process the first generation still took 62.8 s, then 2.516 s warm. This
only pays back for roughly 150 or more same-shape images per process, so the
app uses eager execution by default. The compiled test image looked like the
eager image on visual inspection of this prompt.

Synthetic BF16 attention comparison on this GPU (4096 tokens, 24 heads,
128 head dimensions, 1 batch, 5 warm-ups and 20 CUDA-event timings):

| PyTorch attention kernel | Median |
| --- | ---: |
| SDPA auto | 0.735 ms |
| FlashAttention | 0.667 ms |
| Memory efficient | 1.441 ms |
| Math | 14.218 ms |

SDPA auto also took 0.091 ms at 1024 tokens. The app keeps SDPA's automatic
selection because forcing Flash can reject some mask or shape cases. These
are attention-only timings, not whole-image timings. An attempted build of
official SageAttention 2 failed because the Nix CUDA compiler reports 12.9
while PyTorch uses CUDA 13.0; a separate CUDA 13.0 NVCC wheel could not run on
NixOS without ELF loader patching. SageAttention is not installed. The app's
`auto` attention setting would select it if a compatible build were installed.

Reproduce the measurements with `bench-generation.py` and
`bench-attention.py`. Results and PNG/JSON sidecars are saved under the
Git-ignored `bench/` and `outputs/` directories:

```sh
./run-nixos.sh .venv/bin/python bench-generation.py --preset fast4 --width 1024 --height 1024 --repeats 3
./run-nixos.sh .venv/bin/python bench-generation.py --preset turbo8 --steps 8 --width 1024 --height 1024 --repeats 3
./run-nixos.sh .venv/bin/python bench-generation.py --preset fast4 --width 2048 --height 2048 --vae-tiling-threshold 2049 --repeats 1
./run-nixos.sh .venv/bin/python bench-attention.py --backend sdpa --sdpa-kernel flash
```

GPU driver access, official model loading, 1024², 1536² and 2048² generation,
and the repository tests were verified on 2026-09-22. These visual findings
come from one prompt; other prompts may favor Turbo 8.

Krea 2 Studio is a local browser GUI and independent Diffusers generation engine for Krea 2. It never imports, starts, or sends requests to ComfyUI. Model files can be collected under this project's private `models/` directory and are excluded from Git.

## Start

1. Run `setup-nixos.sh` using the Nix command above. It creates this project's own `.venv` and installs the pinned CUDA 13.0 runtime, Diffusers, and the web API dependencies.
2. Run `./run-nixos.sh`. The app opens at `http://127.0.0.1:8189`.
3. Choose a model and click **モデルを読み込む**. This allocates the selected model in GPU memory; changing the dropdown alone does not load it.
4. Select ordered LoRAs and adjust their strengths. When a model is loaded, LoRA changes are applied automatically through the serialized engine queue.
5. Enter a prompt, choose Turbo 8 or Turbo 4, and generate.
6. To create a high-resolution version, select a completed image and run the separate high-resolution job.

The server binds only to loopback and disables model downloads. Keep machine-specific paths in the private, Git-ignored `config.local.toml`, using `config.example.toml` as a template. Point its component and LoRA paths at the copied files under `models/`. Additional models must explicitly declare `family = "turbo"` or `family = "raw"`; self-contained local Diffusers folders are supported.

## Local model support

- The downloaded official Diffusers Turbo model is configured as the default. The engine also supports a BF16 single-file transformer with separate Qwen3-VL and VAE files.
- The downloaded 4-step distillation LoRA and any added style LoRAs are converted to Diffusers adapters. Adapter order and strengths are recorded in every result.
- ComfyUI ConvRot INT8 checkpoints are listed as unavailable with the exact reason. Their quantization layout is not silently interpreted as BF16.
- An UltraSharp model can be used through Spandrel for high-resolution generation. The current setup uses Lanczos because that upscaler file is absent.
- Raw generation remains disabled until a compatible BF16 or self-contained Diffusers raw model is registered.

Outputs are lossless PNG files under `outputs/YYYY-MM-DD`. Generation settings are stored in PNG iTXt, standard EXIF UserComment, and a JSON sidecar. The gallery is rebuilt from those sidecars after restart.

## Presets

- Turbo 8 uses the distilled Turbo model, 8 steps by default, and Krea guidance `0`.
- Turbo 4 automatically loads the local 4-step adapter at strength `1.0`, uses 4 steps, and Krea guidance `0`.
- High resolution is a separate job that takes an already saved project output. It upscales that image and runs a deterministic refinement schedule without repeating base generation. The 4-step adapter is removed while selected style adapters stay active.

When a compatible SageAttention build is installed, it is scoped to the Krea model and preserves text padding masks through key/value compaction. Unsupported kernel cases fall back to PyTorch SDPA and the fallback count is written to metadata. VC Attention is not exposed because no validated independent implementation is installed.
