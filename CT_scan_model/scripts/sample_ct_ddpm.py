"""Sample synthetic cartesian CT images from a trained DDPM.

This script generates images from pure noise, conditioned on the same
categorical/continuous conditions used during training.

Outputs:
- Cartesian PNGs (CARTESIAN_SIZE x CARTESIAN_SIZE)

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; \
  python -m CT_scan_model.scripts.sample_ct_ddpm \
    --ckpt runs/ct_scan_model/<timestamp>/checkpoint_best.pt \
    --outdir runs/ct_scan_model/<timestamp>/samples \
    --n 8 \
    --cell-format 18650 \
    --manufacturer EVE \
    --chemistry Lithium-ion \
    --slice-depth-relative 0.50 \
    --r-valid-rel 0.94 \
    --cfg-scale 1.0
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


def _build_circle_mask(n: int, size: int, r_valid_rel: float, device: torch.device) -> tuple[torch.Tensor, float]:
    r = float(r_valid_rel) * (float(size) / 2.0)
    r = max(1.0, min(float(size) / 2.0, r))

    cx = (float(size) - 1.0) / 2.0
    cy = (float(size) - 1.0) / 2.0
    ys = torch.arange(size, device=device, dtype=torch.float32)[:, None]
    xs = torch.arange(size, device=device, dtype=torch.float32)[None, :]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    mask2d = (dist2 <= (r**2)).to(dtype=torch.float32)
    x_mask = mask2d[None, None, :, :].expand(int(n), 1, size, size).contiguous()
    return x_mask, r


def _apply_padding_projection(x_img: torch.Tensor, x_mask: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """Project invalid/padded pixels (mask==0) to a fixed training padding value.

    Training uses image=0 outside the circle mask.
    This projection helps prevent drift during sampling.
    """
    pad = torch.tensor(float(pad_value), device=x_img.device, dtype=x_img.dtype)
    return x_img * x_mask + pad * (1.0 - x_mask)


## NOTE: In the cartesian approach we do not remap from polar.


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sample cartesian CT DDPM images from noise.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--n", type=int, required=True, help="Number of samples")

    # Conditions (strict)
    p.add_argument("--cell-format", required=True)
    p.add_argument("--manufacturer", required=True)
    p.add_argument("--chemistry", required=True)
    p.add_argument("--slice-depth-relative", type=float, required=True)
    p.add_argument("--r-valid-rel", type=float, required=True)

    # Sampling
    p.add_argument(
        "--sampler",
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="Sampling method: 'ddpm' (default, full steps) or 'ddim' (fewer steps).",
    )
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Enable ancestral DDPM noise during reverse steps (default: deterministic).",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help=(
            "Noise scale for stochastic sampling. 0.0 = deterministic, 1.0 = full DDPM noise. "
            "Only used when --stochastic is set."
        ),
    )
    p.add_argument(
        "--ddim-steps",
        type=int,
        default=200,
        help="Number of DDIM steps (used only when --sampler=ddim). Typical: 50-200.",
    )
    p.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM noise parameter eta. 0.0 = deterministic, >0 adds noise (used only with --sampler=ddim).",
    )
    p.add_argument(
        "--pad-value",
        type=float,
        default=0.0,
        help="Padding value in model space for mask==0. Default 0.0 matches training padding.",
    )
    p.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA")

    # Output
    p.add_argument("--save-cartesian", action="store_true", default=True)
    p.add_argument("--no-save-cartesian", action="store_false", dest="save_cartesian")
    p.add_argument("--save-mask", action="store_true", help="Save the mask as PNG")

    args = p.parse_args(argv)

    if int(args.n) < 1:
        raise SystemExit("--n must be >= 1")
    if not (0.0 <= float(args.slice_depth_relative) <= 1.0):
        raise SystemExit("--slice-depth-relative must be in [0,1]")
    if not (0.0 < float(args.r_valid_rel) <= 1.0):
        raise SystemExit("--r-valid-rel must be in (0,1]")
    # (no cartesian size/margin args in this approach)
    if float(args.eta) < 0:
        raise SystemExit("--eta must be >= 0")
    if int(args.ddim_steps) < 1:
        raise SystemExit("--ddim-steps must be >= 1")
    if float(args.ddim_eta) < 0:
        raise SystemExit("--ddim-eta must be >= 0")

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
        [float(args.slice_depth_relative), float(args.r_valid_rel)],
        dtype=torch.float32,
        device=device,
    )[None, :].repeat(int(args.n), 1)
    cond = (cat, cont)

    # Load checkpoint.
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet_conditional_cartesian().to(device)
    key = "model" if bool(args.no_ema) else "ema_model"
    if key not in ckpt:
        raise SystemExit(f"Checkpoint missing key '{key}'. Available keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt[key])
    model.eval()

    diffusion = Diffusion(noise_steps=int(args.noise_steps), beta_start=float(args.beta_start), beta_end=float(args.beta_end)).to(device)
    size = int(getattr(project_config, "CARTESIAN_SIZE", 1024))

    # Build fixed circular mask from r-valid-rel.
    x_mask, r_px = _build_circle_mask(int(args.n), size, float(args.r_valid_rel), device=device)

    if bool(args.save_mask):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to save mask PNG.")
        mask_np = (x_mask[0, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(args.outdir, "mask.png"), mask_np)

    with torch.no_grad():
        x_img = torch.randn((int(args.n), 1, size, size), device=device)
        # Ensure padded region matches training representation from the start.
        x_img = _apply_padding_projection(x_img, x_mask, pad_value=float(args.pad_value))

        if str(args.sampler).lower() == "ddim":
            # DDIM sampling: use a subsequence of timesteps, ending at t=1.
            num_steps = int(args.ddim_steps)
            skip = max(1, int(args.noise_steps) // num_steps)
            seq = list(range(1, int(args.noise_steps), skip))
            if seq[-1] != int(args.noise_steps) - 1:
                seq.append(int(args.noise_steps) - 1)
            if seq[0] != 1:
                seq.insert(0, 1)

            for si in range(len(seq) - 1, 0, -1):
                t_i = int(seq[si])
                t_next = int(seq[si - 1])

                t = torch.full((int(args.n),), t_i, device=device, dtype=torch.long)
                model_in = torch.cat([x_img, x_mask], dim=1)
                pred = model(model_in, t, cond)
                if float(args.cfg_scale) != 1.0:
                    uncond = model(model_in, t, None)
                    pred = uncond + float(args.cfg_scale) * (pred - uncond)

                alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]
                t_next_tensor = torch.full((int(args.n),), t_next, device=device, dtype=torch.long)
                alpha_hat_next = diffusion.alpha_hat[t_next_tensor][:, None, None, None]

                # Predict x0 from x_t and eps.
                x0 = (x_img - torch.sqrt(1.0 - alpha_hat_t) * pred) / torch.sqrt(alpha_hat_t)

                eta = float(args.ddim_eta)
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_hat_next) / (1.0 - alpha_hat_t) * (1.0 - alpha_hat_t / alpha_hat_next)
                )
                if eta > 0.0:
                    noise = torch.randn_like(x_img)
                else:
                    noise = torch.zeros_like(x_img)

                x_img = (
                    torch.sqrt(alpha_hat_next) * x0
                    + torch.sqrt(torch.clamp(1.0 - alpha_hat_next - sigma**2, min=0.0)) * pred
                    + sigma * noise
                )

                # Re-project padding each step to prevent drift and boundary leakage.
                x_img = _apply_padding_projection(x_img, x_mask, pad_value=float(args.pad_value))

        else:
            # DDPM sampling (full schedule).
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
                if bool(args.stochastic) and i > 1:
                    noise = float(args.eta) * torch.randn_like(x_img)
                else:
                    noise = torch.zeros_like(x_img)
                x_img = (1.0 / torch.sqrt(alpha)) * (x_img - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred) + torch.sqrt(beta) * noise

                # Re-project padding each step to prevent drift and boundary leakage.
                x_img = _apply_padding_projection(x_img, x_mask, pad_value=float(args.pad_value))

    # Convert to numpy in [-1,1]
    cartesian_samples = x_img[:, 0].detach().cpu().numpy().astype(np.float32)

    if bool(args.save_cartesian):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to save PNG outputs.")
        for i in range(cartesian_samples.shape[0]):
            out = _to_uint8(cartesian_samples[i])
            cv2.imwrite(os.path.join(args.outdir, f"cartesian_{i:03d}.png"), out)

    # Write metadata
    meta = {
        "ckpt": args.ckpt,
        "use_ema": not bool(args.no_ema),
        "device": str(device),
        "n": int(args.n),
        "seed": int(args.seed),
        "cfg_scale": float(args.cfg_scale),
        "sampler": str(args.sampler),
        "stochastic": bool(args.stochastic),
        "eta": float(args.eta),
        "ddim_steps": int(args.ddim_steps),
        "ddim_eta": float(args.ddim_eta),
        "pad_value": float(args.pad_value),
        "noise_steps": int(args.noise_steps),
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "cartesian_size": int(size),
        "r_px": float(r_px),
        "conditions": {
            "cell_format": cell_format,
            "manufacturer": manufacturer,
            "chemistry": chemistry,
            "slice_depth_relative": float(args.slice_depth_relative),
            "r_valid_rel": float(args.r_valid_rel),
        },
        "cat_ids": cat_ids,
        "cont": [float(args.slice_depth_relative), float(args.r_valid_rel)],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote samples to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
