#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if (( $# == 0 )); then
  set -- .venv/bin/krea2-studio
fi
gcc_lib="$(nix eval --raw /etc/nixos#nixosConfigurations.noxos.pkgs.stdenv.cc.cc.lib)"
cuda_nvcc="$(nix eval --raw /etc/nixos#nixosConfigurations.noxos.pkgs.cudaPackages.cuda_nvcc)"
export LD_LIBRARY_PATH="${gcc_lib}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
export TRITON_PTXAS_BLACKWELL_PATH="${cuda_nvcc}/bin/ptxas"
export TRITON_PTXAS_PATH="${cuda_nvcc}/bin/ptxas"
exec nix develop /etc/nixos#cuda -c "$@"
