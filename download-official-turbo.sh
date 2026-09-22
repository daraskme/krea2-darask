#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  echo 'Log in locally first: .venv/bin/hf auth login' >&2
  exit 1
fi

model_dir="$PWD/models/krea2-turbo-diffusers"
.venv/bin/hf download krea/Krea-2-Turbo \
  --revision 98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7 \
  --include 'model_index.json' \
  --include 'scheduler/*' \
  --include 'text_encoder/*' \
  --include 'tokenizer/*' \
  --include 'transformer/*' \
  --include 'vae/*' \
  --local-dir "$model_dir"

if [[ ! -e config.local.toml ]]; then
  cat > config.local.toml <<'EOF'
[engine]
default_model = "krea2-turbo-official"
attention_backend = "sdpa"

[[models]]
id = "krea2-turbo-official"
name = "Krea 2 Turbo (official Diffusers)"
kind = "diffusers"
path = "models/krea2-turbo-diffusers"
family = "turbo"
distilled = true
EOF
fi

echo 'Official Turbo model downloaded. Run: ./run-nixos.sh'
