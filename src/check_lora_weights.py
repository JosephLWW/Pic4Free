#!/usr/bin/env python3
"""LoRA weight audit: Did they learn anything or remain ~zero from initialization?

PEFT inicializa lora_B = 0: un An untrained LoRA has EXACTLY zero effect.
Reports per file: tensor count, fraction of lora_B with norm ~0,
mean lora_A/B norm and top norms. CPU, without GPU.
"""
import argparse
import json
import os
import sys


def audit(path):
    import torch
    from safetensors.torch import load_file

    sd = load_file(path, device="cpu")
    a_norms, b_norms = [], []
    for k, v in sd.items():
        f = v.float()
        n = float(torch.norm(f))
        if k.endswith(".lora_A.weight"):
            a_norms.append(n)
        elif k.endswith(".lora_B.weight"):
            b_norms.append(n)
    import numpy as np

    a = np.array(a_norms)
    b = np.array(b_norms)
    return {
        "file": path,
        "tensors": len(sd),
        "lora_A": {"n": len(a), "mean": float(a.mean()), "max": float(a.max())} if len(a) else None,
        "lora_B": {
            "n": len(b),
            "mean": float(b.mean()),
            "max": float(b.max()),
            "frac_near_zero": float((b < 1e-6).mean()),
        } if len(b) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"[check] torch unavailable: {e}", flush=True)
        return 2

    report = {}
    for p in args.weights:
        if not os.path.isfile(p):
            report[p] = {"error": "does not exist"}
            continue
        try:
            r = audit(p)
        except Exception as e:
            r = {"error": str(e)}
        report[p] = r
        print(f"[check] {p}: {json.dumps(r)}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[check] JSON in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
