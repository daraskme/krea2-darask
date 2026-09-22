#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/hf download lvladikov/Krea2-Turbo-Distill-4step-LoRA \
  krea2_turbo_4step_rank_64_lora.safetensors \
  --revision 597eb1382f58a1fa38b5694ee19a364ee2690103 \
  --local-dir models/loras/krea2
echo '4-step LoRA downloaded.'
