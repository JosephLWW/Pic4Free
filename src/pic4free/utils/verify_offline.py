#!/usr/bin/env python3
"""Offline multi-face re-evaluation of already-generated results (CPU, without GPU).

For each run/task: detects up to 2 faces in the final PNG and computes the matrix
cosine matrix against references A and B SEPARATELY (no averaged), with
greedy assignment. Prints a table and saves JSON.

Usage:
    python -m pic4free.utils.verify_offline --runs data/output/run_X ... \
        --a-dir data/identity_person_A --b-dir data/identity_person_B \
        --tasks 0 53 --out data/output/verify_summary.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def area(f):
    x1, y1, x2, y2 = (float(v) for v in f.bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def normed(e):
    v = np.asarray(e, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def faces_of(app, path, top_k=2):
    img = Image.open(path).convert("RGB")
    faces = sorted(app.get(np.array(img)), key=area, reverse=True)[: max(1, top_k)]
    out = []
    for f in faces:
        e = getattr(f, "normed_embedding", None)
        if e is None:
            continue
        out.append({"bbox": [float(v) for v in f.bbox], "emb": normed(e)})
    return out


def dir_mean(app, d):
    vecs, n = [], 0
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p) or os.path.splitext(name)[1].lower() not in EXTS:
            continue
        n += 1
        fs = faces_of(app, p, top_k=1)
        if fs:
            vecs.append(fs[0]["emb"])
    if not vecs:
        raise RuntimeError(f"No faces in {d} ({n} images vistas).")
    m = np.mean(vecs, axis=0)
    return m / (np.linalg.norm(m) + 1e-12), len(vecs), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=2)
    args = ap.parse_args()

    print("[verify] inicializando InsightFace (CPU)...", flush=True)
    app = load_app()
    ref_a, n_a, seen_a = dir_mean(app, args.a_dir)
    ref_b, n_b, seen_b = dir_mean(app, args.b_dir)
    refs = {"A": ref_a, "B": ref_b}
    print(f"[verify] refs: A={n_a}/{seen_a} faces, B={n_b}/{seen_b} faces.", flush=True)

    summary = {"refs": {"A": {"faces": n_a, "seen": seen_a}, "B": {"faces": n_b, "seen": seen_b}}, "runs": {}}
    for run, task in zip(args.runs, args.tasks if len(args.tasks) > 1 else [args.tasks[0]] * len(args.runs)):
        final = os.path.join(run, "restored", f"task_{task}_final.png")
        entry = {"final": final, "faces": [], "assignment": {}}
        if not os.path.isfile(final):
            entry["note"] = "without PNG final"
            summary["runs"][os.path.basename(run)] = entry
            continue
        faces = faces_of(app, final, top_k=args.top_k)
        for i, f in enumerate(faces):
            entry["faces"].append(
                {"idx": i, "bbox": f["bbox"], "cosines": {k: float(f["emb"] @ r) for k, r in refs.items()}}
            )
        pairs = sorted(
            ((f["idx"], k, c) for f in entry["faces"] for k, c in f["cosines"].items()),
            key=lambda t: t[2],
            reverse=True,
        )
        used_f, used_r = set(), set()
        for fi, k, c in pairs:
            if fi in used_f or k in used_r:
                continue
            entry["assignment"][str(fi)] = {"ref": k, "cosine": c}
            used_f.add(fi)
            used_r.add(k)
        summary["runs"][os.path.basename(run)] = entry
        print(f"[verify] {os.path.basename(run)} task={task}:", flush=True)
        for f in entry["faces"]:
            print(f"  face {f['idx']}: " + " ".join(f"{k}={c:.3f}" for k, c in f["cosines"].items()), flush=True)
        for fi, a in entry["assignment"].items():
            print(f"  -> face {fi} assigned a {a['ref']} (cos={a['cosine']:.3f})", flush=True)

    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[verify] JSON in {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
