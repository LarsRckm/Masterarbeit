"""Sample synthetic polar CT images from a trained polar DDPM.

Generates images from pure noise, conditioned on the categorical/continuous
conditions used during training.  Outputs both the raw polar image and a
back-projected cartesian image.

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; `
  python -m CT_scan_model.scripts.sample_ct_ddpm_polar `
    --ckpt     runs/ct_scan_model_polar/<timestamp>/checkpoint_best.pt `
    --outdir   runs/ct_scan_model_polar/<timestamp>/samples `
    --n        4 `
    --cell-format    18650 `
    --manufacturer   EVE `
    --chemistry      Lithium-ion `
    --slice-depth    0.5 `
    --r-valid-rel    0.85 `
    --cfg-scale      1.0
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
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

from ..diffusion_cartesian import Diffusion
from ..modules_cartesian_ct import UNet_conditional_cartesian
from ..polar_transform import polar_to_cart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _closest(value: str, vocab: List[str]) -> List[str]:
    lower_to_orig = {v.lower(): v for v in vocab}
    matches = difflib.get_close_matches(value.lower(), list(lower_to_orig), n=3, cutoff=0.6)
    return [lower_to_orig[m] for m in matches]


def _require_in_vocab(arg_name: str, value: str, vocab: List[str]) -> str:
    if value in vocab:
        return value
    suggestions = _closest(value, vocab)
    msg = f"Invalid --{arg_name} '{value}'. Allowed: {vocab}."
    if suggestions:
        msg += f" Closest: {suggestions}"
    raise SystemExit(msg)


def _vocab_index(vocab: List[str], value: str) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        raise SystemExit(f"Value '{value}' not in vocab: {vocab}")


def _to_uint8_np(img: np.ndarray) -> np.ndarray:
    """float32 [-1,1] → uint8 [0,255]"""
    return np.clip((img + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _build_mask(n: int, N_r: int, N_theta: int, r_valid_row: int, device: torch.device) -> torch.Tensor:
    """Build a uniform (circular) padding mask: 1 inside r_valid_row, 0 outside."""
    mask = torch.zeros((n, 1, N_r, N_theta), device=device, dtype=torch.float32)
    mask[:, :, :r_valid_row, :] = 1.0
    return mask


def _build_radial_map(n: int, N_r: int, N_theta: int, r_valid_row: int, device: torch.device) -> torch.Tensor:
    """Build normalised radial position map: i/(N_r-1) inside r_valid_row, 0 outside."""
    rows = torch.arange(N_r, device=device, dtype=torch.float32) / max(1.0, float(N_r - 1))
    radial = rows[None, None, :, None].expand(n, 1, N_r, N_theta).clone()
    radial[:, :, r_valid_row:, :] = 0.0
    return radial


@torch.no_grad()
def _sample_ddpm(
    model: torch.nn.Module,
    diffusion: Diffusion,
    cond,
    n: int,
    N_r: int,
    N_theta: int,
    r_valid_row: int,
    pad_value: float,
    cfg_scale: float,
    device: torch.device,
    mask: torch.Tensor = None,
    radial_map: torch.Tensor = None,
) -> torch.Tensor:
    """Full DDPM reverse process.

    The mask (1=valid, 0=padding) is concatenated to the noisy image at every
    step so the model can condition on the cell boundary.  If not provided, a
    uniform circular mask is built from r_valid_row.
    """
    if mask is None:
        mask = _build_mask(n, N_r, N_theta, r_valid_row, device)
    if radial_map is None:
        radial_map = _build_radial_map(n, N_r, N_theta, r_valid_row, device)

    x = torch.randn((n, 1, N_r, N_theta), device=device)
    x[:, :, r_valid_row:, :] = pad_value

    for i in reversed(range(1, diffusion.noise_steps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        model_in = torch.cat([x, mask, radial_map], dim=1)   # [n, 3, N_r, N_theta]
        pred = model(model_in, t, cond)
        if float(cfg_scale) != 1.0:
            uncond = model(model_in, t, None)
            pred = uncond + float(cfg_scale) * (pred - uncond)

        alpha     = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta      = diffusion.beta[t][:, None, None, None]
        noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
        x = (
            (1.0 / torch.sqrt(alpha))
            * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred)
            + torch.sqrt(beta) * noise
        )
        x[:, :, r_valid_row:, :] = pad_value

    return x   # [n, 1, N_r, N_theta] — only the image channel


@torch.no_grad()
def _sample_ddim(
    model: torch.nn.Module,
    diffusion: Diffusion,
    cond,
    n: int,
    N_r: int,
    N_theta: int,
    r_valid_row: int,
    pad_value: float,
    cfg_scale: float,
    device: torch.device,
    ddim_steps: int,
    ddim_eta: float,
    mask: torch.Tensor = None,
    radial_map: torch.Tensor = None,
) -> torch.Tensor:
    """DDIM reverse process (fewer steps).

    mask and radial_map are concatenated to the noisy image at every step.
    """
    if mask is None:
        mask = _build_mask(n, N_r, N_theta, r_valid_row, device)
    if radial_map is None:
        radial_map = _build_radial_map(n, N_r, N_theta, r_valid_row, device)

    skip = max(1, diffusion.noise_steps // ddim_steps)
    seq = list(range(1, diffusion.noise_steps, skip))
    if seq[-1] != diffusion.noise_steps - 1:
        seq.append(diffusion.noise_steps - 1)

    x = torch.randn((n, 1, N_r, N_theta), device=device)
    x[:, :, r_valid_row:, :] = pad_value

    for si in range(len(seq) - 1, 0, -1):
        t_i    = seq[si]
        t_next = seq[si - 1]

        t = torch.full((n,), t_i, device=device, dtype=torch.long)
        model_in = torch.cat([x, mask, radial_map], dim=1)   # [n, 3, N_r, N_theta]
        pred = model(model_in, t, cond)
        if float(cfg_scale) != 1.0:
            uncond = model(model_in, t, None)
            pred = uncond + float(cfg_scale) * (pred - uncond)

        alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]
        t_next_t    = torch.full((n,), t_next, device=device, dtype=torch.long)
        alpha_hat_s = diffusion.alpha_hat[t_next_t][:, None, None, None]

        x0 = (x - torch.sqrt(1.0 - alpha_hat_t) * pred) / torch.sqrt(alpha_hat_t).clamp(min=1e-8)
        sigma = ddim_eta * torch.sqrt(
            (1.0 - alpha_hat_s) / (1.0 - alpha_hat_t).clamp(min=1e-8)
            * (1.0 - alpha_hat_t / alpha_hat_s.clamp(min=1e-8))
        )
        noise = torch.randn_like(x) if ddim_eta > 0.0 else torch.zeros_like(x)
        x = (
            torch.sqrt(alpha_hat_s) * x0
            + torch.sqrt(torch.clamp(1.0 - alpha_hat_s - sigma ** 2, min=0.0)) * pred
            + sigma * noise
        )
        x[:, :, r_valid_row:, :] = pad_value

    return x   # [n, 1, N_r, N_theta] — only the image channel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sample polar CT DDPM images from a trained checkpoint.")

    p.add_argument("--ckpt",   required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--n",      type=int, required=True, help="Number of samples to generate")

    # Conditioning
    p.add_argument("--cell-format",  required=True,        help=f"One of: {project_config.CELL_FORMAT_VOCAB}")
    p.add_argument("--manufacturer", required=True,        help=f"One of: {project_config.MANUFACTURER_VOCAB}")
    p.add_argument("--chemistry",    required=True,        help=f"One of: {project_config.CHEMISTRY_VOCAB}")
    p.add_argument("--slice-depth",  type=float, required=True, help="Relative slice depth in [0, 1]")
    p.add_argument("--r-valid-rel",  type=float, required=True,
                   help="max(r_valid_per_angle) / image_half_size in (0, 1]")

    # Sampler
    p.add_argument("--sampler",     choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--ddim-steps",  type=int,   default=200, help="DDIM steps (only with --sampler=ddim)")
    p.add_argument("--ddim-eta",    type=float, default=0.0, help="DDIM eta: 0=deterministic, 1=full noise")
    p.add_argument("--cfg-scale",   type=float, default=1.0, help="Classifier-free guidance scale")
    p.add_argument("--noise-steps", type=int,   default=1000)
    p.add_argument("--beta-schedule", choices=["linear", "cosine"], default="cosine",
                   help="Must match the schedule used during training of --ckpt.")
    p.add_argument("--beta-start",  type=float, default=1e-4)
    p.add_argument("--beta-end",    type=float, default=0.02)

    # Misc
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA")
    p.add_argument("--cart-size", type=int, default=None,
                   help="Output size for back-projected cartesian image (default: CARTESIAN_SIZE from config)")

    args = p.parse_args(argv)

    # --- Validate ---
    if int(args.n) < 1:
        raise SystemExit("--n must be >= 1")
    if not (0.0 <= float(args.slice_depth) <= 1.0):
        raise SystemExit("--slice-depth must be in [0, 1]")
    if not (0.0 < float(args.r_valid_rel) <= 1.0):
        raise SystemExit("--r-valid-rel must be in (0, 1]")

    cell_format  = _require_in_vocab("cell-format",  args.cell_format,  project_config.CELL_FORMAT_VOCAB)
    manufacturer = _require_in_vocab("manufacturer", args.manufacturer, project_config.MANUFACTURER_VOCAB)
    chemistry    = _require_in_vocab("chemistry",    args.chemistry,    project_config.CHEMISTRY_VOCAB)

    # --- Config ---
    N_r       = int(getattr(project_config, "POLAR_N_R",       512))
    N_theta   = int(getattr(project_config, "POLAR_N_THETA",   1024))
    pad_value = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))
    cart_size = int(args.cart_size or getattr(project_config, "CARTESIAN_SIZE", 1024))

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Polar dims: N_r={N_r}, N_theta={N_theta}, pad_value={pad_value}")

    os.makedirs(args.outdir, exist_ok=True)

    # --- Conditioning tensors ---
    cat_ids = [
        _vocab_index(project_config.CELL_FORMAT_VOCAB,  cell_format),
        _vocab_index(project_config.MANUFACTURER_VOCAB, manufacturer),
        _vocab_index(project_config.CHEMISTRY_VOCAB,    chemistry),
    ]
    cat  = torch.tensor(cat_ids, dtype=torch.long,    device=device)[None].repeat(int(args.n), 1)
    cont = torch.tensor(
        [float(args.slice_depth), float(args.r_valid_rel)],
        dtype=torch.float32, device=device,
    )[None].repeat(int(args.n), 1)
    cond = (cat, cont)

    # --- Load checkpoint ---
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    key  = "model" if bool(args.no_ema) else "ema_model"
    if key not in ckpt:
        raise SystemExit(f"Key '{key}' not found in checkpoint. Available: {list(ckpt.keys())}")

    model = UNet_conditional_cartesian(
        c_in=int(getattr(project_config, "POLAR_UNET_IN_CHANNELS", 1)),
        num_downs=int(getattr(project_config, "UNET_NUM_DOWNS_POLAR", 5)),
        base_channels=int(getattr(project_config, "POLAR_UNET_BASE_CHANNELS",
                                   project_config.UNET_BASE_CHANNELS)),
    ).to(device)
    model.load_state_dict(ckpt[key])
    model.eval()
    print(f"Loaded {'EMA' if not args.no_ema else 'raw'} weights from: {args.ckpt}")

    diffusion = Diffusion(
        noise_steps=int(args.noise_steps),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        schedule=str(args.beta_schedule),
    ).to(device)

    # --- Row cutoff + mask from r_valid_rel ---
    r_valid_row = int(round(float(args.r_valid_rel) * (N_r - 1)))
    print(f"r_valid_row: {r_valid_row} / {N_r}  (r_valid_rel={args.r_valid_rel:.4f})")

    # Uniform circular mask + radial map built from r_valid_row.
    mask       = _build_mask(      int(args.n), N_r, N_theta, r_valid_row, device)
    radial_map = _build_radial_map(int(args.n), N_r, N_theta, r_valid_row, device)

    # --- Sample ---
    print(f"Sampling {args.n} image(s) with {args.sampler.upper()}...")
    if args.sampler == "ddim":
        samples = _sample_ddim(
            model, diffusion, cond,
            n=int(args.n), N_r=N_r, N_theta=N_theta,
            r_valid_row=r_valid_row, pad_value=pad_value,
            cfg_scale=float(args.cfg_scale), device=device,
            ddim_steps=int(args.ddim_steps), ddim_eta=float(args.ddim_eta),
            mask=mask, radial_map=radial_map,
        )
    else:
        samples = _sample_ddpm(
            model, diffusion, cond,
            n=int(args.n), N_r=N_r, N_theta=N_theta,
            r_valid_row=r_valid_row, pad_value=pad_value,
            cfg_scale=float(args.cfg_scale), device=device,
            mask=mask, radial_map=radial_map,
        )

    # samples: [n, 1, N_r, N_theta] on device
    samples_np = samples[:, 0].cpu().numpy()  # [n, N_r, N_theta]

    if cv2 is None:
        raise SystemExit("OpenCV (cv2) is required to save PNG outputs.")

    cx = cy = float(cart_size) / 2.0
    r_max = float(args.r_valid_rel) * cx  # scale for back-projection

    for i in range(int(args.n)):
        polar_np = samples_np[i]  # [N_r, N_theta]

        # Save polar image.
        polar_u8 = _to_uint8_np(np.clip(polar_np, -1.0, 1.0))
        polar_path = os.path.join(args.outdir, f"sample_{i:03d}_polar.png")
        cv2.imwrite(polar_path, polar_u8)

        # Back-project to cartesian.
        cart_np = polar_to_cart(polar_np, cx, cy, r_max, N_r, N_theta, cart_size, pad_value)
        cart_u8 = _to_uint8_np(np.clip(cart_np, -1.0, 1.0))
        cart_path = os.path.join(args.outdir, f"sample_{i:03d}_cart.png")
        cv2.imwrite(cart_path, cart_u8)

        print(f"  [{i+1}/{args.n}]  polar → {polar_path}")
        print(f"  [{i+1}/{args.n}]  cart  → {cart_path}")

    # --- Metadata ---
    meta = {
        "ckpt": args.ckpt,
        "use_ema": not bool(args.no_ema),
        "device": str(device),
        "n": int(args.n),
        "seed": int(args.seed),
        "sampler": args.sampler,
        "ddim_steps": int(args.ddim_steps),
        "ddim_eta": float(args.ddim_eta),
        "cfg_scale": float(args.cfg_scale),
        "noise_steps": int(args.noise_steps),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "N_r": N_r,
        "N_theta": N_theta,
        "pad_value": pad_value,
        "r_valid_row": r_valid_row,
        "cart_size": cart_size,
        "conditions": {
            "cell_format": cell_format,
            "manufacturer": manufacturer,
            "chemistry": chemistry,
            "slice_depth": float(args.slice_depth),
            "r_valid_rel": float(args.r_valid_rel),
        },
    }
    meta_path = os.path.join(args.outdir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Wrote {args.n} sample(s) to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
