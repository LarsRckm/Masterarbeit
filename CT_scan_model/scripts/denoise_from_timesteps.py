"""Denoising trajectory test starting from discrete timesteps.

Analogous to `model_debug_memorization/scripts/denoise_from_timesteps.py`.

Loads a trained checkpoint (EMA recommended), takes one real image sample from
the provided index/splits, adds forward diffusion noise at user-specified
discrete timesteps, and then runs the reverse process from each timestep down
to t=1 using the model.

Outputs per timestep:
  - noised_tXXXX.png
  - denoised_from_tXXXX.png
  - metadata_tXXXX.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from ..dataset_ct_polar import BatteryCTSelectedSamplesDataset
from ..diffusion_polar import Diffusion
from ..modules_polar_ct import UNet_conditional_polar


def _to_uint8(img: torch.Tensor) -> torch.Tensor:
    """Convert float [-1,1] tensor to uint8 [0,255]."""
    return ((img + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)


def _parse_timesteps(s: str) -> list[int]:
    items = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
    out: list[int] = []
    for it in items:
        out.append(int(it))
    # keep order but remove duplicates
    seen = set()
    uniq: list[int] = []
    for t in out:
        if t in seen:
            continue
        uniq.append(t)
        seen.add(t)
    return uniq


def _select_one_sample(index_path: str, splits_path: str, split_name: str, cell_id: Optional[str]) -> dict:
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    cell_ids = list(splits.get(f"{split_name}_cells", []))
    if not cell_ids:
        raise RuntimeError(f"No cells found for split='{split_name}'.")

    if cell_id is None:
        cid = str(cell_ids[0])
    else:
        cid = str(cell_id)
        if cid not in set(cell_ids):
            raise RuntimeError(f"Requested cell_id not in split '{split_name}': {cid}")

    info = index.get("cells", {}).get(cid)
    if not info:
        raise RuntimeError(f"cell_id not found in index: {cid}")
    imgs = list(info.get("images", []))
    if not imgs:
        raise RuntimeError(f"No images listed for cell_id: {cid}")

    mid = min(imgs, key=lambda d: abs(float(d.get("rel_depth", 0.0)) - 0.5))
    return {"cell_id": cid, "relpath": mid["relpath"], "rel_depth": float(mid.get("rel_depth", 0.0))}


@torch.no_grad()
def _reverse_from_xt(
    diffusion: Diffusion,
    model: torch.nn.Module,
    x_img: torch.Tensor,
    x_mask: torch.Tensor,
    cond,
    start_t: int,
    device: torch.device,
    cfg_scale: float = 1.0,
    stochastic: bool = False,
) -> torch.Tensor:
    """Run reverse steps from start_t down to 1.

    Args:
        x_img:  [1,1,R,Theta] at timestep start_t
        x_mask: [1,1,R,Theta]
        cond:   (cat, cont)
    """
    model.eval()
    x = x_img
    for i in reversed(range(1, int(start_t) + 1)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        model_in = torch.cat([x, x_mask], dim=1)
        pred = model(model_in, t, cond)
        if float(cfg_scale) != 1.0:
            uncond = model(model_in, t, None)
            pred = uncond + float(cfg_scale) * (pred - uncond)

        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
        if stochastic and i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (1.0 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred
        ) + torch.sqrt(beta) * noise

    model.train()
    return x


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Denoise a real CT polar image starting from discrete timesteps")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint_best.pt (or any checkpoint)")
    p.add_argument("--index", required=True, help="Index JSON used for selecting the image")
    p.add_argument("--geometry", required=True, help="Geometry JSON used for selecting the image")
    p.add_argument("--splits", required=True, help="Splits JSON used for selecting the image")
    p.add_argument("--split", default="train", choices=["train", "val", "test"], help="Which split to draw from")
    p.add_argument("--cell-id", default=None, help="Specific cell_id to use (must be in the chosen split)")
    p.add_argument("--timesteps", required=True, help="Comma-separated timesteps, e.g. '10,100,500,999'")
    p.add_argument("--outdir", default=None, help="Output directory (default next to ckpt)")
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--stochastic", action="store_true", help="Add DDPM noise during reverse steps (default: deterministic)")
    p.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA")

    # Must match training diffusion schedule.
    p.add_argument("--noise-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)
    args = p.parse_args(argv)

    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to write PNG outputs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = _parse_timesteps(args.timesteps)
    if not timesteps:
        raise ValueError("--timesteps must contain at least one integer")

    diffusion = Diffusion(
        noise_steps=int(args.noise_steps),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet_conditional_polar().to(device)
    key = "model" if bool(args.no_ema) else "ema_model"
    if key not in ckpt:
        raise RuntimeError(f"Checkpoint missing key '{key}'. Available keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt[key])
    model.eval()

    sample = _select_one_sample(args.index, args.splits, split_name=str(args.split), cell_id=args.cell_id)
    ds = BatteryCTSelectedSamplesDataset(args.index, args.geometry, samples=[sample])
    x, cond, mask = ds[0]
    x = x[None, ...].to(device)  # [1,2,R,Theta]
    cat, cont = cond
    cat = cat[None, ...].to(device)
    cont = cont[None, ...].to(device)
    cond_batched = (cat, cont)
    x_mask = mask[None, None, ...].to(device=device, dtype=torch.float32)

    if args.outdir is None:
        base = os.path.dirname(os.path.abspath(args.ckpt))
        outdir = os.path.join(base, "denoise_timesteps")
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    for t0 in timesteps:
        if not (1 <= int(t0) < int(args.noise_steps)):
            raise ValueError(f"Invalid timestep {t0}. Must be in [1, noise_steps-1].")

        t = torch.tensor([int(t0)], device=device, dtype=torch.long)
        x_t, _eps = diffusion.noise_images(x, t)
        x_img_t = x_t[:, :1]

        den = _reverse_from_xt(
            diffusion=diffusion,
            model=model,
            x_img=x_img_t,
            x_mask=x_mask,
            cond=cond_batched,
            start_t=int(t0),
            device=device,
            cfg_scale=float(args.cfg_scale),
            stochastic=bool(args.stochastic),
        )

        noised_u8 = _to_uint8(x_img_t[0, 0].detach().cpu())
        den_u8 = _to_uint8(den[0, 0].detach().cpu())
        cv2.imwrite(os.path.join(outdir, f"noised_t{int(t0):04d}.png"), noised_u8.numpy())
        cv2.imwrite(os.path.join(outdir, f"denoised_from_t{int(t0):04d}.png"), den_u8.numpy())

        meta = {
            "ckpt": os.path.abspath(args.ckpt),
            "use_ema": not bool(args.no_ema),
            "index": os.path.abspath(args.index),
            "geometry": os.path.abspath(args.geometry),
            "splits": os.path.abspath(args.splits),
            "split": str(args.split),
            "sample": sample,
            "timestep_start": int(t0),
            "cfg_scale": float(args.cfg_scale),
            "stochastic": bool(args.stochastic),
            "noise_steps": int(args.noise_steps),
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
        }
        with open(os.path.join(outdir, f"metadata_t{int(t0):04d}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"Wrote outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
