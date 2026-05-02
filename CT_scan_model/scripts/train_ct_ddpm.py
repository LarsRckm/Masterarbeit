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
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math

from ..dataset_ct_polar import BatteryCTPerCellDataset, BatteryCTPolarDataset
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
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs. If 0, derive from --slices-per-cell and batch size.",
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

    # Validation/test still evaluate on all slices (as currently indexed).
    val_ds = BatteryCTPolarDataset(args.index, args.geometry, splits_json=args.splits, split="val")
    test_ds = BatteryCTPolarDataset(args.index, args.geometry, splits_json=args.splits, split="test")

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
        losses = []
        with torch.no_grad():
            for x, cond, mask in loader:
                x = x.to(device)
                mask = mask.to(device)
                cat, cont = cond
                cat = cat.to(device)
                cont = cont.to(device)
                t = diffusion.sample_timesteps(x.shape[0], device=device)
                x_t, noise = diffusion.noise_images(x, t)
                pred = net(x_t, t, (cat, cont))
                losses.append(float(masked_mse(pred, noise, mask).item()))
        net.train()
        return float(sum(losses) / max(1, len(losses)))

    # Training
    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        # Ensure per-cell slice selection changes each epoch.
        train_ds.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        for step, (x, cond, mask) in enumerate(train_loader, start=1):
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

        train_loss = running / max(1, len(train_loader))
        val_loss = run_eval(val_loader, use_ema=True)
        print(f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | val_loss(ema)={val_loss:.6f}")

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
            ckpt_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch:04d}.pt")
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
