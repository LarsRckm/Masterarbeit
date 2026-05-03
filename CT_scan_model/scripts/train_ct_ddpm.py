"""Training script for polar CT DDPM with train/val/test cell splits.

This script trains UNet_conditional_polar on polar tensors [image, mask].
It uses a masked MSE loss so padded pixels do not dominate training.

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; \
  python -m model.CT_scan_model.scripts.train_ct_ddpm \
    --index    model/CT_scan_model/cell_index.json \
    --geometry model/CT_scan_model/cell_geometry.json \
    --splits   model/CT_scan_model/splits.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import warnings
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

import math

from ..dataset_ct_polar import BatteryCTPerCellDataset, BatteryCTSelectedSamplesDataset
from ..diffusion_polar import Diffusion
from ..modules_polar_ct import UNet_conditional_polar


class EMA:
    def __init__(self, beta: float = 0.995):
        self.beta = float(beta)
        self.step = 0

    @torch.no_grad()
    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 0):
        if self.step < step_start_ema:
            ema_model.load_state_dict(model.state_dict())
            self.step += 1
            return
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.beta).add_(p.data, alpha=1.0 - self.beta)
        self.step += 1


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: [B, R, Theta] or [B,1,R,Theta]
    if mask.dim() == 3:
        mask = mask[:, None, :, :]
    mask = mask.to(dtype=pred.dtype)
    num = ((pred - target) ** 2 * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


def _to_uint8(img: torch.Tensor) -> torch.Tensor:
    """Convert float [-1,1] tensor to uint8 [0,255]."""
    return ((img + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def sample_with_mask(
    diffusion: Diffusion,
    model: nn.Module,
    n: int,
    cond,
    mask: torch.Tensor,
    device: torch.device,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """DDPM sampling with a fixed padding mask.

    Returns image channel only: [n, 1, R, Theta]
    """
    model.eval()
    r = int(mask.shape[-2])
    th = int(mask.shape[-1])
    x_img = torch.randn((n, 1, r, th), device=device)
    x_mask = mask.to(device=device, dtype=torch.float32)
    if x_mask.dim() == 2:
        x_mask = x_mask[None, None, :, :]
    elif x_mask.dim() == 3:
        x_mask = x_mask[:, None, :, :]
    if x_mask.shape[0] == 1 and n != 1:
        x_mask = x_mask.expand(n, -1, -1, -1)

    for i in reversed(range(1, diffusion.noise_steps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        model_in = torch.cat([x_img, x_mask], dim=1)
        pred = model(model_in, t, cond)
        if float(cfg_scale) != 1.0:
            uncond = model(model_in, t, None)
            pred = uncond + float(cfg_scale) * (pred - uncond)

        alpha = diffusion.alpha[t][:, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
        beta = diffusion.beta[t][:, None, None, None]
        noise = torch.randn_like(x_img) if i > 1 else torch.zeros_like(x_img)
        x_img = (1.0 / torch.sqrt(alpha)) * (x_img - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred) + torch.sqrt(beta) * noise

    model.train()
    return x_img


def _seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train polar CT DDPM (single GPU).")
    p.add_argument("--index", default=os.path.join("model", "CT_scan_model", "cell_index.json"))
    p.add_argument("--geometry", default=os.path.join("model", "CT_scan_model", "cell_geometry.json"))
    p.add_argument("--splits", default=os.path.join("model", "CT_scan_model", "splits.json"))

    p.add_argument(
        "--slices-per-cell",
        type=int,
        default=1,
        help="How many distinct slice depths per cell should be seen over the full training run.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs. If 0, derive from --slices-per-cell (training only).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--noise-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--ema-beta", type=float, default=0.995)
    p.add_argument("--p-uncond", type=float, default=0.1, help="Probability to drop conditioning (CFG training)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default=None, help="Output directory (default: model/CT_scan_model/runs/<timestamp>)")
    p.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument(
        "--tqdm",
        action="store_true",
        help="Show a per-epoch tqdm progress bar (useful for debugging long epochs).",
    )

    # Optional qualitative sampling during training
    p.add_argument(
        "--sample-every",
        type=int,
        default=0,
        help="If >0, generate qualitative samples every N epochs.",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=2,
        help="How many samples to generate per validation condition.",
    )
    p.add_argument(
        "--sample-cfg-scale",
        type=float,
        default=1.0,
        help="CFG scale used for qualitative sampling.",
    )
    args = p.parse_args(argv)

    _seed_everything(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.run_dir is None:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("runs", "ct_scan_model", ts)
    else:
        run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    weights_dir = os.path.join(run_dir, "weights")
    val_pictures_dir = os.path.join(run_dir, "val_pictures")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(val_pictures_dir, exist_ok=True)

    # Save run config.
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if int(args.slices_per_cell) < 1:
        raise ValueError("--slices-per-cell must be >= 1")

    # Cell-level training dataset (one sample per cell per epoch).
    train_ds = BatteryCTPerCellDataset(
        args.index,
        args.geometry,
        splits_json=args.splits,
        split="train",
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        pad_to_batch=True,
    )

    # Validation/test: small cell-format coverage subset.
    # We pick 1 cell per format and a mid-depth slice for each.
    def _select_mid_slice_per_format(split_name: str) -> list[dict]:
        with open(args.splits, "r", encoding="utf-8") as f:
            splits = json.load(f)
        with open(args.index, "r", encoding="utf-8") as f:
            index = json.load(f)

        allowed_cells = set(splits[f"{split_name}_cells"])
        # Build cell -> (format, images)
        cell_infos = {}
        for cid, info in index["cells"].items():
            if cid in allowed_cells:
                cell_infos[cid] = info

        chosen = []
        for fmt in ["18650", "2170", "4680"]:
            # choose first cell that matches format
            candidates = [cid for cid, info in cell_infos.items() if str(info.get("cell_format", "")) == fmt]
            if not candidates:
                continue
            cid = sorted(candidates)[0]
            imgs = list(cell_infos[cid].get("images", []))
            if not imgs:
                continue
            # pick mid depth (closest to 0.5)
            mid = min(imgs, key=lambda d: abs(float(d.get("rel_depth", 0.0)) - 0.5))
            chosen.append({"cell_id": cid, "relpath": mid["relpath"], "rel_depth": float(mid.get("rel_depth", 0.0))})
        return chosen

    val_samples = _select_mid_slice_per_format("val")
    test_samples = _select_mid_slice_per_format("test")
    if not val_samples:
        raise RuntimeError("No validation samples selected. Check that val split contains 18650/2170/4680 cells.")
    if not test_samples:
        raise RuntimeError("No test samples selected. Check that test split contains 18650/2170/4680 cells.")

    val_ds = BatteryCTSelectedSamplesDataset(args.index, args.geometry, samples=val_samples)
    test_ds = BatteryCTSelectedSamplesDataset(args.index, args.geometry, samples=test_samples)

    # Derive epoch count if not explicitly set.
    if int(args.epochs) <= 0:
        # Epoch = one pass over (padded) train cells. We cycle one slice per cell
        # per epoch, so seeing K distinct slices per cell implies K epochs.
        args.epochs = int(args.slices_per_cell)
        print(
            "Derived epochs:",
            args.epochs,
            f"(epoch=pass_over_cells, train_cells_per_epoch={len(train_ds)}, slices_per_cell={int(args.slices_per_cell)})",
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet_conditional_polar().to(device)
    ema_model = UNet_conditional_polar().to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p_ in ema_model.parameters():
        p_.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    ema = EMA(beta=float(args.ema_beta))
    diffusion = Diffusion(noise_steps=int(args.noise_steps), beta_start=float(args.beta_start), beta_end=float(args.beta_end)).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def run_eval(loader: DataLoader, use_ema: bool = False) -> float:
        net = ema_model if use_ema else model
        net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x, cond, mask in loader:
                bs = int(x.shape[0])
                x = x.to(device)
                mask = mask.to(device)
                cat, cont = cond
                cat = cat.to(device)
                cont = cont.to(device)
                t = diffusion.sample_timesteps(bs, device=device)
                x_t, noise = diffusion.noise_images(x, t)
                pred = net(x_t, t, (cat, cont))
                loss = masked_mse(pred, noise, mask).item()
                total += float(loss) * bs
                count += bs
        net.train()
        return float(total / max(1, count))

    # Training
    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        # Ensure per-cell slice selection changes each epoch.
        train_ds.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0

        use_tqdm = (bool(args.tqdm) or sys.stderr.isatty()) and tqdm is not None
        if bool(args.tqdm) and tqdm is None:
            raise RuntimeError("tqdm is not installed but --tqdm was requested. Install via: pip install tqdm")

        train_iter = train_loader
        pbar = None
        if use_tqdm:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{int(args.epochs)}", unit="batch", leave=False)
            train_iter = pbar

        for step, (x, cond, mask) in enumerate(train_iter, start=1):
            x = x.to(device)
            mask = mask.to(device)
            cat, cont = cond
            cat = cat.to(device)
            cont = cont.to(device)

            # CFG training: randomly drop conditioning.
            if float(args.p_uncond) > 0 and random.random() < float(args.p_uncond):
                cond_in = None
            else:
                cond_in = (cat, cont)

            t = diffusion.sample_timesteps(x.shape[0], device=device)
            x_t, noise = diffusion.noise_images(x, t)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(x_t, t, cond_in)
                loss = masked_mse(pred, noise, mask)
                loss = loss / max(1, int(args.accumulation_steps))

            scaler.scale(loss).backward()

            if step % max(1, int(args.accumulation_steps)) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.step_ema(ema_model, model, step_start_ema=0)

            running += float(loss.item())

            if pbar is not None:
                # Show the *unscaled* (per-step) loss.
                pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

        train_loss = running / max(1, len(train_loader))
        val_loss = run_eval(val_loader, use_ema=True)
        print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss(ema)={val_loss:.6f}")

        # Save validation pictures (polar images) for the selected validation samples.
        # We save the un-noised input image channel for quick visual sanity checks.
        if cv2 is not None:
            try:
                with torch.no_grad():
                    for j, (vx, _vcond, _vmask) in enumerate(val_loader):
                        imgs = vx[:, 0].cpu().numpy()
                        for k in range(imgs.shape[0]):
                            img = ((imgs[k] + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
                            outp = os.path.join(val_pictures_dir, f"epoch_{epoch:04d}_val_{j:02d}_{k:02d}.png")
                            cv2.imwrite(outp, img)
            except Exception:
                pass

        # Qualitative sampling (generate synthetic samples for fixed validation conditions).
        if int(args.sample_every) > 0 and (epoch % int(args.sample_every) == 0):
            if cv2 is None:
                warnings.warn("cv2 not available; skipping qualitative sampling.")
            else:
                samples_dir = os.path.join(run_dir, "samples_training", f"epoch_{epoch:04d}")
                os.makedirs(samples_dir, exist_ok=True)
                try:
                    with torch.no_grad():
                        ema_model.eval()
                        # Use the selected validation samples (one per format) as fixed conditions.
                        for ci, (vx, vcond, vmask) in enumerate(val_loader):
                            cat, cont = vcond
                            cat = cat.to(device)
                            cont = cont.to(device)
                            cond_fixed = (
                                cat[:1].expand(int(args.sample_n), -1),
                                cont[:1].expand(int(args.sample_n), -1),
                            )
                            gen = sample_with_mask(
                                diffusion=diffusion,
                                model=ema_model,
                                n=int(args.sample_n),
                                cond=cond_fixed,
                                mask=vmask[:1],
                                device=device,
                                cfg_scale=float(args.sample_cfg_scale),
                            )
                            gen_u8 = _to_uint8(gen[:, 0].cpu())
                            for si in range(gen_u8.shape[0]):
                                outp = os.path.join(samples_dir, f"cond_{ci:02d}_sample_{si:02d}.png")
                                cv2.imwrite(outp, gen_u8[si].numpy())

                        # Write sampling metadata once per epoch.
                        meta = {
                            "epoch": int(epoch),
                            "sample_every": int(args.sample_every),
                            "sample_n": int(args.sample_n),
                            "sample_cfg_scale": float(args.sample_cfg_scale),
                            "checkpoint_best_path": os.path.join(run_dir, "checkpoint_best.pt"),
                            "val_selected_samples": val_samples,
                        }
                        with open(os.path.join(samples_dir, "metadata.json"), "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)
                except Exception as e:
                    warnings.warn(f"Qualitative sampling failed: {e}")

        # Save best + periodic checkpoints
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                },
                ckpt_path,
            )
        if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
            ckpt_path = os.path.join(weights_dir, f"checkpoint_epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                },
                ckpt_path,
            )

    test_loss = run_eval(test_loader, use_ema=True)
    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test_loss_ema": test_loss, "best_val_ema": best_val}, f, indent=2)
    print(f"Test loss (EMA): {test_loss:.6f}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    p.add_argument(
        "--slices-per-cell",
        type=int,
        default=1,
        help="How many distinct slice depths per cell should be seen over the full training run.",
    )
