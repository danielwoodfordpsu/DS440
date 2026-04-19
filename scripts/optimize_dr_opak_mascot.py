#!/usr/bin/env python3
"""Resize mascot for web; prefer WebP (loaded first by app)."""

from __future__ import annotations

import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
SRC = os.path.join(ASSETS, "dr_opak_mascot.png")
OUT_WEBP = os.path.join(ASSETS, "dr_opak_mascot.webp")
OUT_PNG = os.path.join(ASSETS, "dr_opak_mascot.png")
MAX_EDGE = 400


def main() -> int:
    if not os.path.isfile(SRC):
        print(f"Missing {SRC}", file=sys.stderr)
        return 1
    img = Image.open(SRC).convert("RGBA")
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        if w >= h:
            nw, nh = MAX_EDGE, max(1, int(h * MAX_EDGE / w))
        else:
            nw, nh = max(1, int(w * MAX_EDGE / h)), MAX_EDGE
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    img.save(OUT_WEBP, "WEBP", quality=82, method=6)
    img.save(OUT_PNG, "PNG", optimize=True)
    for label, path in ("WebP", OUT_WEBP), ("PNG", OUT_PNG):
        print(f"{label}: {os.path.getsize(path) // 1024} KB -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
