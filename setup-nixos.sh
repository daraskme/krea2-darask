#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
  echo 'uv is required; run: nix shell nixpkgs#uv nixpkgs#python312 -c ./setup-nixos.sh' >&2
  exit 1
fi
if [[ -d .venv && -f .venv/bin/hf &&
      "$(head -n 1 .venv/bin/hf)" != "#!$(pwd)/.venv/bin/python" ]]; then
  uv venv --python 3.12 --clear .venv
elif [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements-nixos.txt
uv pip install --python .venv/bin/python -e . --no-deps
./run-nixos.sh .venv/bin/python -c 'import torch; print("PyTorch", torch.__version__, "CUDA available:", torch.cuda.is_available())'
echo 'Run: ./run-nixos.sh'
