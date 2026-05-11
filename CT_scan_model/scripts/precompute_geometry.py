"""Precompute per-cell geometry (center + usable radius) for CT training.

For each cell_id in the cell_index.json, this script:
- selects a representative slice (rel_depth closest to 0.5)
- estimates (cx, cy, r_valid) via Otsu + morphology + enclosing circle
- stores metadata (manufacturer, chemistry, voxel size)

Output JSON is consumed by the dataset and build_splits.py.

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; \
  python -m CT_scan_model.scripts.precompute_geometry \
    --index CT_scan_model/cell_index.json \
    --out   CT_scan_model/cell_geometry.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None

try:
    from model import battery_metadata
except ImportError:  # pragma: no cover
    import battery_metadata  # type: ignore


def _load_gray(path: str) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for geometry precomputation.")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img


def _estimate_circle(gray: np.ndarray) -> Tuple[float, float, float]:
    """Return (cx, cy, r) using a robust, simple heuristic."""
    h, w = gray.shape[:2]
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
    r0 = min(h, w) / 2.0 - 2.0

    if cv2 is None:
        return cx0, cy0, max(1.0, r0)

    try:
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((7, 7), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cx0, cy0, max(1.0, r0)

        largest = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(largest)
        # Sanity clamp.
        r = float(np.clip(r, 1.0, min(h, w) / 2.0))
        return float(cx), float(cy), float(r)
    except Exception:
        return cx0, cy0, max(1.0, r0)


def _pick_representative(images: List[dict], target_rel_depth: float = 0.5) -> dict:
    # images: [{relpath, rel_depth, abs_depth}, ...]
    return min(images, key=lambda d: abs(float(d.get("rel_depth", 0.0)) - target_rel_depth))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Precompute per-cell CT geometry (cx,cy,r_valid).")
    p.add_argument(
        "--index",
        default=os.path.join("CT_scan_model", "cell_index.json"),
        help="Input cell_index.json path",
    )
    p.add_argument(
        "--out",
        default=os.path.join("CT_scan_model", "cell_geometry.json"),
        help="Output geometry JSON path",
    )
    p.add_argument("--target-rel-depth", type=float, default=0.5)
    args = p.parse_args(argv)

    with open(args.index, "r", encoding="utf-8") as f:
        index = json.load(f)

    base_path = index["base_path"]
    cells: Dict[str, dict] = index["cells"]

    geom: Dict[str, dict] = {}
    for cell_id, cell_info in cells.items():
        images = cell_info.get("images", [])
        if not images:
            continue
        rep = _pick_representative(images, target_rel_depth=float(args.target_rel_depth))
        rep_relpath = rep["relpath"]
        rep_abspath = os.path.join(base_path, rep_relpath)

        gray = _load_gray(rep_abspath)
        h, w = gray.shape[:2]
        cx, cy, r_valid = _estimate_circle(gray)

        # Derive metadata.
        cell_format = cell_info.get("cell_format", "Unknown")
        # manufacturer inference uses image size and (optionally) relative path hints.
        manufacturer = battery_metadata.determine_manufacturer(w, h, rep_relpath)
        chemistry = battery_metadata.determine_chemistry(manufacturer)
        voxel_size_um = battery_metadata.determine_voxel_size(cell_format)

        geom[cell_id] = {
            "rep_image": rep_relpath,
            "cx": float(cx),
            "cy": float(cy),
            "r_valid": float(r_valid),
            "width": int(w),
            "height": int(h),
            "cell_format": cell_format,
            "manufacturer": manufacturer,
            "chemistry": chemistry,
            "voxel_size_um": None if voxel_size_um is None else float(voxel_size_um),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"base_path": base_path, "cells": geom}, f, indent=2)

    print(f"Wrote geometry: {args.out}  (cells={len(geom)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
