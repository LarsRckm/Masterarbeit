"""Sample synthetic polar CT images from a trained DDPM.

This script generates images from pure noise, conditioned on the same
categorical/continuous conditions used during training.

Outputs:
- Polar PNGs (R x Theta)
- Optional Cartesian PNGs via inverse remap (centered)

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; \
  python -m model.CT_scan_model.scripts.sample_ct_ddpm \
    --ckpt runs/ct_scan_model/<timestamp>/checkpoint_best.pt \
    --outdir runs/ct_scan_model/<timestamp>/samples \
    --n 8 \
    --cell-format 18650 \
    --manufacturer EVE \
    --chemistry Lithium-ion \
    --slice-depth-relative 0.50 \
    --voxel-size-um 14.4 \
    --r-valid-rel 0.94 \
    --cfg-scale 1.0
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import random
import sys
import warnings
from typing import List, Optional

import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import config as project_config
except ImportError:  # pragma: no cover
    import config as project_config  # type: ignore

from ..diffusion_polar import Diffusion
from ..modules_polar_ct import UNet_conditional_polar


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _closest_matches_case_insensitive(value: str, vocab: List[str], n: int = 3, cutoff: float = 0.6) -> List[str]:
    lower_to_orig = {}
    vocab_lower = []
    for v in vocab:
        vl = v.lower()
        if vl not in lower_to_orig:
            lower_to_orig[vl] = v
        vocab_lower.append(vl)
    matches = difflib.get_close_matches(value.lower(), vocab_lower, n=n, cutoff=cutoff)
    # keep order, unique
    out = []
    seen = set()
    for m in matches:
        orig = lower_to_orig.get(m, m)
        if orig not in seen:
            out.append(orig)
            seen.add(orig)
    return out


def _require_in_vocab(arg_name: str, value: str, vocab: List[str]) -> str:
    if value in vocab:
        return value
    suggestions = _closest_matches_case_insensitive(value, vocab)
    msg = f"Invalid --{arg_name} '{value}'. Allowed: {vocab}."
    if suggestions:
        msg += f" Closest: {suggestions}"
    raise SystemExit(msg)


def _vocab_index(vocab: List[str], value: str) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        # Should not happen with strict validation.
        raise SystemExit(f"Value '{value}' not in vocab: {vocab}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    # img in [-1, 1]
    x = (img + 1.0) * 0.5 * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def _build_polar_mask(n: int, r_model: int, theta_bins: int, r_valid_rel: float, device: torch.device) -> tuple[torch.Tensor, int]:
    r_use = int(round(float(r_valid_rel) * float(r_model)))
    r_use = max(1, min(int(r_model), r_use))
    x_mask = torch.zeros((n, 1, r_model, theta_bins), device=device, dtype=torch.float32)
    x_mask[:, :, :r_use, :] = 1.0
    return x_mask, r_use


def _build_inverse_remap_maps(size: int, r_use_effective: int, theta_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for cartesian output via remap.")

    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0

    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dx = X - cx
    dy = Y - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    theta = np.where(theta < 0, theta + 2.0 * np.pi, theta)

    map_x = (theta / (2.0 * np.pi)) * float(theta_bins)
    map_x = np.mod(map_x, float(theta_bins)).astype(np.float32)
    map_y = r.astype(np.float32)

    outside = r > float(r_use_effective)
    map_x[outside] = -1.0
    map_y[outside] = -1.0
    return map_x, map_y


def _polar_to_cartesian_remap(polar: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for cartesian output via remap.")
    return cv2.remap(
        polar.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sample polar CT DDPM images from noise.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--n", type=int, required=True, help="Number of samples")

    # Conditions (strict)
    p.add_argument("--cell-format", required=True)
    p.add_argument("--manufacturer", required=True)
    p.add_argument("--chemistry", required=True)
    p.add_argument("--slice-depth-relative", type=float, required=True)
    p.add_argument("--voxel-size-um", type=float, required=True)
    p.add_argument("--r-valid-rel", type=float, required=True)

    # Sampling
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)
    p.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA")

    # Output
    p.add_argument("--save-polar", action="store_true", default=True)
    p.add_argument("--no-save-polar", action="store_false", dest="save_polar")
    p.add_argument("--save-cartesian", action="store_true", default=True)
    p.add_argument("--no-save-cartesian", action="store_false", dest="save_cartesian")
    p.add_argument("--cartesian-size", default="auto", help="'auto' or integer size")
    p.add_argument("--cartesian-margin", type=int, default=4)
    p.add_argument("--save-mask", action="store_true", help="Save the polar mask as PNG")

    args = p.parse_args(argv)

    if int(args.n) < 1:
        raise SystemExit("--n must be >= 1")
    if not (0.0 <= float(args.slice_depth_relative) <= 1.0):
        raise SystemExit("--slice-depth-relative must be in [0,1]")
    if not (0.0 < float(args.r_valid_rel) <= 1.0):
        raise SystemExit("--r-valid-rel must be in (0,1]")
    if float(args.voxel_size_um) <= 0:
        raise SystemExit("--voxel-size-um must be > 0")
    if int(args.cartesian_margin) < 0:
        raise SystemExit("--cartesian-margin must be >= 0")

    cell_format = _require_in_vocab("cell-format", str(args.cell_format), project_config.CELL_FORMAT_VOCAB)
    manufacturer = _require_in_vocab("manufacturer", str(args.manufacturer), project_config.MANUFACTURER_VOCAB)
    chemistry = _require_in_vocab("chemistry", str(args.chemistry), project_config.CHEMISTRY_VOCAB)

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _ensure_dir(args.outdir)
    # Save metadata early.
    meta_path = os.path.join(args.outdir, "metadata.json")

    # Build cond tensors.
    cat_ids = [
        _vocab_index(project_config.CELL_FORMAT_VOCAB, cell_format),
        _vocab_index(project_config.MANUFACTURER_VOCAB, manufacturer),
        _vocab_index(project_config.CHEMISTRY_VOCAB, chemistry),
    ]
    cat = torch.tensor(cat_ids, dtype=torch.long, device=device)[None, :].repeat(int(args.n), 1)
    cont = torch.tensor(
        [float(args.slice_depth_relative), float(args.voxel_size_um), float(args.r_valid_rel)],
        dtype=torch.float32,
        device=device,
    )[None, :].repeat(int(args.n), 1)
    cond = (cat, cont)

    # Load checkpoint.
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet_conditional_polar().to(device)
    key = "model" if bool(args.no_ema) else "ema_model"
    if key not in ckpt:
        raise SystemExit(f"Checkpoint missing key '{key}'. Available keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt[key])
    model.eval()

    # Diffusion schedule.
    diffusion = Diffusion(noise_steps=int(args.noise_steps), beta_start=float(args.beta_start), beta_end=float(args.beta_end)).to(device)
    r_model = int(project_config.POLAR_R_MODEL)
    theta_bins = int(project_config.POLAR_THETA_BINS)

    # Build fixed mask from r-valid-rel.
    x_mask, r_use = _build_polar_mask(int(args.n), r_model, theta_bins, float(args.r_valid_rel), device=device)

    if bool(args.save_mask):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to save mask PNG.")
        mask_np = (x_mask[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(args.outdir, "mask.png"), mask_np)

    # Sampling loop (DDPM).
    with torch.no_grad():
        x_img = torch.randn((int(args.n), 1, r_model, theta_bins), device=device)
        for i in reversed(range(1, int(args.noise_steps))):
            t = torch.full((int(args.n),), i, device=device, dtype=torch.long)
            model_in = torch.cat([x_img, x_mask], dim=1)
            pred = model(model_in, t, cond)
            if float(args.cfg_scale) != 1.0:
                uncond = model(model_in, t, None)
                pred = uncond + float(args.cfg_scale) * (pred - uncond)

            alpha = diffusion.alpha[t][:, None, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
            beta = diffusion.beta[t][:, None, None, None]
            noise = torch.randn_like(x_img) if i > 1 else torch.zeros_like(x_img)
            x_img = (1.0 / torch.sqrt(alpha)) * (x_img - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred) + torch.sqrt(beta) * noise

    # Convert to numpy in [-1,1]
    polar_samples = x_img[:, 0].detach().cpu().numpy().astype(np.float32)

    # Prepare cartesian mapping if needed.
    cartesian_size = None
    r_use_effective = r_use
    map_x = map_y = None
    if bool(args.save_cartesian):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for cartesian output.")
        margin = int(args.cartesian_margin)

        if str(args.cartesian_size).lower() == "auto":
            cartesian_size = int(math.ceil(2.0 * float(r_use)) + 2 * margin)
        else:
            try:
                cartesian_size = int(args.cartesian_size)
            except ValueError:
                raise SystemExit("--cartesian-size must be 'auto' or an integer")

        if cartesian_size <= 0:
            raise SystemExit("--cartesian-size must be > 0")

        r_max_allowed = int(cartesian_size // 2 - margin)
        if r_max_allowed < 1:
            raise SystemExit("--cartesian-size too small for the given margin")
        if r_use > r_max_allowed:
            warnings.warn(
                f"cartesian-size={cartesian_size} too small for r_use={r_use}; clamping radius to {r_max_allowed} for cartesian remap.",
                RuntimeWarning,
            )
            r_use_effective = r_max_allowed

        map_x, map_y = _build_inverse_remap_maps(cartesian_size, r_use_effective, theta_bins)

    # Save images.
    if bool(args.save_polar):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to save PNG outputs.")
        for i in range(polar_samples.shape[0]):
            out = _to_uint8(polar_samples[i])
            cv2.imwrite(os.path.join(args.outdir, f"polar_{i:03d}.png"), out)

    if bool(args.save_cartesian):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to save PNG outputs.")
        assert cartesian_size is not None and map_x is not None and map_y is not None
        for i in range(polar_samples.shape[0]):
            cart = _polar_to_cartesian_remap(polar_samples[i], map_x, map_y)
            out = _to_uint8(cart)
            cv2.imwrite(os.path.join(args.outdir, f"cartesian_{i:03d}.png"), out)

    # Write metadata
    meta = {
        "ckpt": args.ckpt,
        "use_ema": not bool(args.no_ema),
        "device": str(device),
        "n": int(args.n),
        "seed": int(args.seed),
        "cfg_scale": float(args.cfg_scale),
        "noise_steps": int(args.noise_steps),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "polar_shape": [r_model, theta_bins],
        "cartesian_size": cartesian_size,
        "cartesian_margin": int(args.cartesian_margin),
        "r_use": int(r_use),
        "r_use_effective_cartesian": int(r_use_effective),
        "conditions": {
            "cell_format": cell_format,
            "manufacturer": manufacturer,
            "chemistry": chemistry,
            "slice_depth_relative": float(args.slice_depth_relative),
            "voxel_size_um": float(args.voxel_size_um),
            "r_valid_rel": float(args.r_valid_rel),
        },
        "cat_ids": cat_ids,
        "cont": [float(args.slice_depth_relative), float(args.voxel_size_um), float(args.r_valid_rel)],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote samples to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
