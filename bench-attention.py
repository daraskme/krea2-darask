"""Synthetic attention throughput check for this workstation; no model weights needed."""
import argparse
import json
import statistics

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sdpa", "sage2"), required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sdpa-kernel", choices=("auto", "flash", "math", "efficient"), default="auto")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(42)
    shape = (1, args.heads, args.tokens, args.head_dim)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    if args.backend == "sage2":
        from sageattention import sageattn
        run = lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        selected = {"flash": SDPBackend.FLASH_ATTENTION,
                    "math": SDPBackend.MATH,
                    "efficient": SDPBackend.EFFICIENT_ATTENTION}.get(args.sdpa_kernel)
        def run():
            if selected is None:
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            with sdpa_kernel(selected):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    print(json.dumps({"backend": args.backend, "gpu": torch.cuda.get_device_name(),
                      "sdpa_kernel": args.sdpa_kernel if args.backend == "sdpa" else None,
                      "tokens": args.tokens, "heads": args.heads, "head_dim": args.head_dim,
                      "median_ms": round(statistics.median(times), 3),
                      "min_ms": round(min(times), 3), "torch": torch.__version__}))


if __name__ == "__main__":
    main()
