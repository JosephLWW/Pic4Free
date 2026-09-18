#!/usr/bin/env python3
"""Phase 0 LoRA curation per person (CPU).

For each identity directory (A/B):
  - Detects faces (InsightFace, the largest face).
  - Skips images without a face (with warning), e.g. 3 known images in A.
  - Crops a square around the face and shoulders (bbox x CROP_EXPAND, clip) + JPEG q95.
  - Writes a .txt caption with trigger (P4F_A / P4F_B).
  - Held-out reservation: the 2 images with the smallest faces (hard cases) for
    checkpoint selection by ArcFace cosine.

Output:
  data/lora/train_{A,B}/, data/lora/heldout_{A,B}/, data/lora/manifest.json

Uso:
    python src/curate_lora.py --a-dir data/identity_person_A \
        --b-dir data/identity_person_B --out data/lora \
        --trigger-a P4F_A --trigger-b P4F_B
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".webp")
CROP_EXPAND = 2.0
JPEG_Q = 95


def load_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def area(f):
    x1, y1, x2, y2 = (float(v) for v in f.bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def curate(app, src_dir, dst_train, dst_held, trigger, held_n=2, freeze=None):
    import shutil

    # Full regeneration: avoids orphaned crops of deleted images.
    for _d in (dst_train, dst_held):
        if os.path.isdir(_d):
            shutil.rmtree(_d)
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_held, exist_ok=True)
    found = []
    seen = 0
    for name in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, name)
        if not os.path.isfile(p) or os.path.splitext(name)[1].lower() not in EXTS:
            continue
        seen += 1
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"  [skip] {name}: no legible ({e})", flush=True)
            continue
        try:
            faces = app.get(np.array(img))
        except Exception as e:
            print(f"  [skip] {name}: detection failed ({e})", flush=True)
            continue
        if not faces:
            print(f"  [skip] {name}: no faces", flush=True)
            continue
        best = max(faces, key=area)
        x1, y1, x2, y2 = (float(v) for v in best.bbox)
        W, H = img.size
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) / 2.0 * CROP_EXPAND
        box = (
            int(max(0, cx - half)),
            int(max(0, cy - half)),
            int(min(W, cx + half)),
            int(min(H, cy + half)),
        )
        if box[2] - box[0] < 64 or box[3] - box[1] < 64:
            print(f"  [skip] {name}: recorte degenerado", flush=True)
            continue
        found.append({"src": name, "crop": img.crop(box), "area": area(best)})
    # Held-out: by default the of face most small (hard cases); if exists
    # previous manifest with --freeze-heldout, is preserve esos files for
    # so cosines are comparable across dataset versions.
    frozen = set(freeze or [])
    if frozen:
        held = [d for d in found if os.path.splitext(d["src"])[0] in frozen]
        missing = frozen - {os.path.splitext(d["src"])[0] for d in found}
        for m in sorted(missing):
            print(f"  [heldout] {m} already unavailable/with face; remains out.", flush=True)
    else:
        found.sort(key=lambda d: d["area"])
        held = found[: min(held_n, max(0, len(found) - 1))]
    held_names = {id(d) for d in held}
    train = [d for d in found if id(d) not in held_names]
    caption = (
        f"{trigger} portrait photo, face and shoulders, natural light, "
        f"photorealistic, sharp facial detail"
    )
    for d, folder in ((train, dst_train), (held, dst_held)):
        for item in d:
            stem = os.path.splitext(item["src"])[0]
            item["crop"].save(os.path.join(folder, f"{stem}.jpg"), quality=JPEG_Q)
            with open(os.path.join(folder, f"{stem}.txt"), "w") as fh:
                fh.write(caption + "\n")
    return {
        "seen": seen,
        "train": len(train),
        "heldout": len(held),
        "heldout_files": sorted(os.path.splitext(d["src"])[0] for d in held),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--b-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trigger-a", default="P4F_A")
    ap.add_argument("--trigger-b", default="P4F_B")
    ap.add_argument("--held-n", type=int, default=2)
    ap.add_argument(
        "--freeze-heldout",
        action="store_true",
        help="Keep heldout from the existing manifest (comparability).",
    )
    args = ap.parse_args()

    print("[curate] InsightFace (CPU)...", flush=True)
    app = load_app()
    prev = {}
    if args.freeze_heldout:
        try:
            with open(os.path.join(args.out, "manifest.json")) as fh:
                old = json.load(fh)
            for label in ("A", "B"):
                prev[label] = set(old.get(label, {}).get("heldout_files", []))
            print(f"[curate] held-out frozen: { {k: sorted(v) for k, v in prev.items()} }", flush=True)
        except Exception as e:
            print(f"[curate] without previous manifest usesble ({e}); new selection.", flush=True)
    manifest = {}
    for label, src, trig in (("A", args.a_dir, args.trigger_a), ("B", args.b_dir, args.trigger_b)):
        print(f"[curate] {label}: {src}", flush=True)
        manifest[label] = curate(
            app,
            src,
            os.path.join(args.out, f"train_{label}"),
            os.path.join(args.out, f"heldout_{label}"),
            trig,
            held_n=args.held_n,
            freeze=prev.get(label),
        )
        print(f"[curate] {label}: {manifest[label]}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[curate] manifest in {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
