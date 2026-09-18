#!/usr/bin/env python3
"""Identity dataset audit (CPU).

For each A/B image, reports:
  - n_faces, relative size of the largest (face area / image area)
  - blur (Laplacian variance, OpenCV) and mean brightness
  - ArcFace cosine vs directory mean (outliers = different person/bad shot)
  - duplicates by perceptual hash (dHash)
Verdict per image: KEEP / REVIEW / DROP + reason.
Filters enabled via CLI (see --help). Does not delete anything; only reports.

Usage:
    python -m pic4free.cli.audit_dataset --a-dir data/identity_person_A \
        --b-dir data/identity_person_B --out data/lora \
        --min-face-frac 0.02 --blur-min 40.0 --cos-min 0.35
"""
import argparse
import hashlib
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


def laplacian_var(gray: np.ndarray) -> float:
    # OpenCV >=5 does not support certain combinations (e.g. float32->float64):
    # any failure falls back to NumPy.
    try:
        import cv2

        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        return float(cv2.Laplacian(g8, cv2.CV_64F).var())
    except Exception:
        gx, gy = np.gradient(np.asarray(gray, dtype=np.float32))
        return float((gx.var() + gy.var()) / 2.0)


def dhash(img: Image.Image, size: int = 8) -> str:
    g = np.array(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.float32)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return f"{h:016x}"


def normed(e) -> np.ndarray:
    v = np.asarray(e, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def audit_dir(app, d, args):
    rows = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p) or os.path.splitext(name)[1].lower() not in EXTS:
            continue
        row = {"file": name, "verdict": "KEEP", "reasons": []}
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            row.update(verdict="DROP", reasons=[f"ilegible: {e}"])
            rows.append(row)
            continue
        arr = np.array(img)
        W, H = img.size
        gray = np.array(img.convert("L"), dtype=np.float32)
        row["brightness"] = round(float(gray.mean()), 1)
        row["blur"] = round(laplacian_var(gray), 1)
        row["dhash"] = dhash(img)
        try:
            faces = app.get(arr)
        except Exception as e:
            row.update(verdict="REVIEW", reasons=[f"detector failure: {e}"])
            rows.append(row)
            continue
        row["n_faces"] = len(faces)
        if not faces:
            row.update(verdict="DROP", reasons=["no faces"])
            rows.append(row)
            continue
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        x1, y1, x2, y2 = (float(v) for v in faces[0].bbox)
        row["face_frac"] = round(max(0.0, x2 - x1) * max(0.0, y2 - y1) / (W * H), 4)
        emb = getattr(faces[0], "normed_embedding", None)
        row["emb"] = normed(emb).tolist() if emb is not None else None
        if emb is None:
            row.update(verdict="REVIEW", reasons=["no embedding"])
        rows.append(row)

    # Second pass: cosine vs mean + duplicates + thresholds.
    embs = [np.array(r["emb"]) for r in rows if r.get("emb") is not None]
    mean = np.mean(embs, axis=0) if embs else None
    if mean is not None:
        mean = mean / (np.linalg.norm(mean) + 1e-12)
    seen_hash = {}
    for r in rows:
        if r["verdict"] == "DROP":
            continue
        if r.get("emb") is not None and mean is not None:
            r["cos_vs_mean"] = round(float(np.dot(np.array(r["emb"]), mean)), 3)
            del r["emb"]
            if r["cos_vs_mean"] < args.cos_min:
                r["verdict"] = "REVIEW"
                r["reasons"].append(f"possible different person/bad shot (cos={r['cos_vs_mean']})")
        elif "emb" in r:
            del r["emb"]
        if r.get("face_frac", 1.0) < args.min_face_frac:
            r["verdict"] = "REVIEW" if r["verdict"] == "KEEP" else r["verdict"]
            r["reasons"].append(f"small face (frac={r['face_frac']})")
        if r.get("blur", 1e9) < args.blur_min:
            r["verdict"] = "REVIEW" if r["verdict"] == "KEEP" else r["verdict"]
            r["reasons"].append(f"blurry (laplaciano={r['blur']})")
        h = r.get("dhash")
        if h in seen_hash:
            r["verdict"] = "DROP" if r["verdict"] != "REVIEW" else r["verdict"]
            r["reasons"].append(f"duplicate of {seen_hash[h]}")
        else:
            seen_hash[h] = r["file"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-face-frac", type=float, default=0.02)
    ap.add_argument("--blur-min", type=float, default=40.0)
    ap.add_argument("--cos-min", type=float, default=0.35)
    args = ap.parse_args()

    print("[audit] InsightFace (CPU)...", flush=True)
    app = load_app()
    report = {}
    for label, d in (("A", args.a_dir), ("B", args.b_dir)):
        print(f"[audit] {label}: {d}", flush=True)
        rows = audit_dir(app, d, args)
        report[label] = rows
        k = sum(1 for r in rows if r["verdict"] == "KEEP")
        rv = sum(1 for r in rows if r["verdict"] == "REVIEW")
        dr = sum(1 for r in rows if r["verdict"] == "DROP")
        print(f"[audit] {label}: {len(rows)} vistas | KEEP={k} REVIEW={rv} DROP={dr}", flush=True)
        for r in rows:
            if r["verdict"] != "KEEP":
                print(f"  [{r['verdict']}] {r['file']}: {'; '.join(r['reasons'])}", flush=True)
    with open(os.path.join(args.out, "audit.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[audit] JSON in {args.out}/audit.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
