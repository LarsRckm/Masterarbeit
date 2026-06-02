"""Training script for polar CT DDPM with train/val/test cell splits.

Polar convention
----------------
Input tensor x: [1, POLAR_N_R, POLAR_N_THETA]  (c_in=1, no mask channel)
  channel 0  polar image, normalised to [-1, 1]

Images are transformed directly from their original resolution (no resize to
CARTESIAN_SIZE).  The radial scale r_max = mean(r_valid_per_angle) is computed
per cell so each cell fills all N_r rows.  Pixels outside the actual boundary
receive pad_value and are excluded from loss via masked_mse.

The UNet architecture is identical to the cartesian model (same channel count,
same num_downs); only the spatial dimensions differ (512×1024 vs 1024×1024).

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; `
  python -m CT_scan_model.scripts.train_ct_ddpm_polar `
    --index    CT_scan_model/cell_index.json `
    --geometry CT_scan_model/cell_geometry.json `
    --splits   CT_scan_model/splits.json
"""

from __future__ import annotations

import argparse
import csv
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

from ..dataset_ct_polar import (
    BatteryCTPolarPerCellDataset,
    BatteryCTPolarSelectedSamplesDataset,
    BatteryCTPolarUniformCellsMaxPicturesDataset,
)
from ..diffusion_cartesian import Diffusion
from ..modules_cartesian_ct import UNet_conditional_cartesian
from ..polar_transform import polar_to_cart
from .sample_ct_ddpm_polar import _sample_ddpm, _sample_ddim, _to_uint8_np

try:
    from model import config as project_config
except ImportError:  # pragma: no cover
    import config as project_config  # type: ignore


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask[:, None, :, :]
    mask = mask.to(dtype=pred.dtype)
    num = ((pred - target) ** 2 * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _append_csv_row(path: str, header: list, row: list) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _polar_sample_to_uint8_cart(
    polar_img: torch.Tensor,
    r_valid_rel: float,
    cart_size: int,
    N_r: int,
    N_theta: int,
    pad_value: float,
) -> "np.ndarray":
    """Convert a single generated polar image tensor to a uint8 cartesian image."""
    import numpy as np
    p = polar_img.cpu().numpy()
    cx = cy = cart_size / 2.0
    r_max = float(r_valid_rel) * (float(cart_size) / 2.0)
    cart = polar_to_cart(p, cx, cy, r_max, N_r, N_theta, cart_size, pad_value)
    return _to_uint8_np(cart)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_eval(
    loader: DataLoader,
    model: nn.Module,
    ema_model: nn.Module,
    diffusion: Diffusion,
    device: torch.device,
    use_ema: bool = False,
) -> float:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Train polar CT DDPM (single GPU).")
    p.add_argument("--index", default=os.path.join("CT_scan_model", "cell_index.json"))
    p.add_argument("--geometry", default=os.path.join("CT_scan_model", "cell_geometry.json"))
    p.add_argument("--splits", default=os.path.join("CT_scan_model", "splits.json"))

    p.add_argument("--slices-per-cell", type=int, default=1)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--max-pictures", type=int, default=0)
    p.add_argument("--val-every-pictures", type=int, default=0)
    p.add_argument("--no-ema-val", action="store_true")

    p.add_argument("--noise-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--ema-beta", type=float, default=0.995)
    p.add_argument("--p-uncond", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--tqdm", action="store_true", default=True)

    p.add_argument("--resume", action="store_true",
                   help="Resume training from a checkpoint in --run-dir")
    p.add_argument("--resume-from", default=None,
                   help="Explicit checkpoint path to resume from (overrides default search)")

    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--sample-n", type=int, default=1)
    p.add_argument("--sample-cfg-scale", type=float, default=1.0)
    p.add_argument("--sample-sampler", choices=["ddpm", "ddim"], default="ddpm",
                   help="Sampler used for qualitative samples during training")
    p.add_argument("--sample-ddim-steps", type=int, default=200,
                   help="DDIM steps (only when --sample-sampler=ddim)")
    p.add_argument("--sample-ddim-eta", type=float, default=0.0,
                   help="DDIM eta: 0=deterministic, 1=full noise")

    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args(argv)

    # Polar config constants.
    N_r = int(getattr(project_config, "POLAR_N_R", 512))
    N_theta = int(getattr(project_config, "POLAR_N_THETA", 1024))
    pad_value = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))
    cart_size = int(getattr(project_config, "CARTESIAN_SIZE", 1024))

    clearml_task = None
    clearml_logger = None
    if not bool(args.no_clearml):
        from clearml import Task  # type: ignore
        clearml_task = Task.init(
            project_name="ISEAnet",
            task_name="tte-lre-master-ct-scan-polar",
            task_type=Task.TaskTypes.training,
        )
        clearml_task.connect(vars(args))
        clearml_logger = clearml_task.get_logger()

    _seed_everything(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Polar dims: N_r={N_r}, N_theta={N_theta}, pad_value={pad_value}")

    if args.run_dir is None:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("runs", "ct_scan_model_polar", ts)
    else:
        run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    weights_dir = os.path.join(run_dir, "weights")
    val_pictures_dir = os.path.join(run_dir, "val_pictures")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(val_pictures_dir, exist_ok=True)

    epoch_loss_csv = os.path.join(run_dir, "loss_per_epoch.csv")
    batch_loss_csv = os.path.join(run_dir, "loss_per_batch.csv")
    val_loss_csv = os.path.join(run_dir, "loss_per_val.csv")

    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if int(args.slices_per_cell) < 1:
        raise ValueError("--slices-per-cell must be >= 1")

    # --- Datasets ---
    if int(args.max_pictures) > 0:
        bs = int(args.batch_size)
        max_pics_rounded = int(((int(args.max_pictures) + bs - 1) // bs) * bs)
        train_ds = BatteryCTPolarUniformCellsMaxPicturesDataset(
            args.index, args.geometry,
            splits_json=args.splits, split="train",
            max_pictures=max_pics_rounded, seed=int(args.seed),
        )
        print(
            f"Picture-budget mode: max_pictures={args.max_pictures} -> rounded={max_pics_rounded} "
            f"(batch_size={bs}, train_cells={len(train_ds.cell_ids)})"
        )
    else:
        train_ds = BatteryCTPolarPerCellDataset(
            args.index, args.geometry,
            splits_json=args.splits, split="train",
            batch_size=int(args.batch_size), seed=int(args.seed), pad_to_batch=True,
        )

    def _select_mid_slice_per_format(split_name: str) -> list:
        with open(args.splits, "r", encoding="utf-8") as f:
            splits = json.load(f)
        with open(args.index, "r", encoding="utf-8") as f:
            index = json.load(f)
        allowed_cells = set(splits[f"{split_name}_cells"])
        cell_infos = {
            cid: info for cid, info in index["cells"].items() if cid in allowed_cells
        }
        chosen = []
        for fmt in ["18650", "2170", "4680"]:
            candidates = [
                cid for cid, info in cell_infos.items()
                if str(info.get("cell_format", "")) == fmt
            ]
            if not candidates:
                continue
            cid = sorted(candidates)[0]
            imgs = list(cell_infos[cid].get("images", []))
            if not imgs:
                continue
            mid = min(imgs, key=lambda d: abs(float(d.get("rel_depth", 0.0)) - 0.5))
            chosen.append({
                "cell_id": cid,
                "relpath": mid["relpath"],
                "rel_depth": float(mid.get("rel_depth", 0.0)),
            })
        return chosen

    val_samples = _select_mid_slice_per_format("val")
    test_samples = _select_mid_slice_per_format("test")
    if not val_samples:
        raise RuntimeError("No validation samples found.")
    if not test_samples:
        raise RuntimeError("No test samples found.")

    val_ds = BatteryCTPolarSelectedSamplesDataset(args.index, args.geometry, samples=val_samples)
    test_ds = BatteryCTPolarSelectedSamplesDataset(args.index, args.geometry, samples=test_samples)

    if int(args.max_pictures) <= 0 and int(args.epochs) <= 0:
        args.epochs = int(args.slices_per_cell)
        print(f"Derived epochs: {args.epochs}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=(int(args.max_pictures) <= 0),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(),
    )

    # --- Model (same architecture as cartesian, different spatial dims) ---
    model = UNet_conditional_cartesian(
        c_in=int(getattr(project_config, "POLAR_UNET_IN_CHANNELS", 2)),
        num_downs=int(getattr(project_config, "UNET_NUM_DOWNS_POLAR", 5)),
        base_channels=int(getattr(project_config, "POLAR_UNET_BASE_CHANNELS", project_config.UNET_BASE_CHANNELS)),
    ).to(device)
    ema_model = UNet_conditional_cartesian(
        c_in=int(getattr(project_config, "POLAR_UNET_IN_CHANNELS", 2)),
        num_downs=int(getattr(project_config, "UNET_NUM_DOWNS_POLAR", 5)),
        base_channels=int(getattr(project_config, "POLAR_UNET_BASE_CHANNELS", project_config.UNET_BASE_CHANNELS)),
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    ema = EMA(beta=float(args.ema_beta))
    diffusion = Diffusion(
        noise_steps=int(args.noise_steps),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
    ).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    start_epoch = 1

    # --- Resume from checkpoint ---
    if bool(args.resume) or args.resume_from is not None:
        if args.resume_from is not None:
            ckpt_path = args.resume_from
        else:
            # Priority: checkpoint_best.pt → latest epoch checkpoint
            ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")
            if not os.path.isfile(ckpt_path):
                import glob as _glob
                epoch_ckpts = sorted(
                    _glob.glob(os.path.join(weights_dir, "checkpoint_epoch_*.pt"))
                )
                if epoch_ckpts:
                    ckpt_path = epoch_ckpts[-1]
                else:
                    raise FileNotFoundError(
                        f"No checkpoint found in {run_dir}. "
                        "Pass --resume-from <path> to specify one explicitly."
                    )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val    = float(ckpt.get("best_val", float("inf")))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"  → continuing from epoch {start_epoch}  (best_val={best_val:.6f})")

        if start_epoch > int(args.epochs) and int(args.max_pictures) <= 0:
            raise ValueError(
                f"Checkpoint epoch ({start_epoch - 1}) >= --epochs ({args.epochs}). "
                "Increase --epochs to continue training."
            )

    # --- Training loop (epoch-based) ---
    if int(args.max_pictures) <= 0:
        global_step = (start_epoch - 1) * len(train_loader)
        for epoch in range(start_epoch, int(args.epochs) + 1):
            train_ds.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.0

            use_tqdm = (bool(args.tqdm) or sys.stderr.isatty()) and tqdm is not None
            train_iter = train_loader
            pbar = None
            if use_tqdm:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{int(args.epochs)}", unit="batch", leave=False)
                train_iter = pbar

            for step, (x, cond, mask) in enumerate(train_iter, start=1):
                global_step += 1
                x = x.to(device)
                mask = mask.to(device)
                cat, cont = cond
                cat = cat.to(device)
                cont = cont.to(device)

                cond_in = None if (float(args.p_uncond) > 0 and random.random() < float(args.p_uncond)) else (cat, cont)

                t = diffusion.sample_timesteps(x.shape[0], device=device)
                x_t, noise = diffusion.noise_images(x, t)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred = model(x_t, t, cond_in)
                    loss_raw = masked_mse(pred, noise, mask)
                    loss = loss_raw / max(1, int(args.accumulation_steps))

                scaler.scale(loss).backward()

                if step % max(1, int(args.accumulation_steps)) == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.step_ema(ema_model, model, step_start_ema=0)

                running += float(loss.item())
                _append_csv_row(
                    batch_loss_csv,
                    header=["epoch", "batch", "train_loss"],
                    row=[epoch, step, float(loss_raw.detach().item())],
                )
                if clearml_logger is not None:
                    clearml_logger.report_scalar("loss", "train", iteration=global_step, value=float(loss_raw.detach().item()))
                if pbar is not None:
                    pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

            train_loss = running / max(1, len(train_loader))
            val_loss = run_eval(val_loader, model, ema_model, diffusion, device, use_ema=(not bool(args.no_ema_val)))
            print(f"Epoch {epoch:04d} | train={train_loss:.6f} | val={'ema' if not args.no_ema_val else 'raw'}={val_loss:.6f}")

            if clearml_logger is not None:
                clearml_logger.report_scalar("loss_epoch", "train", iteration=epoch, value=float(train_loss))
                clearml_logger.report_scalar("loss_epoch", "val", iteration=epoch, value=float(val_loss))

            # Save ground-truth polar images and back-projected cartesian images.
            if cv2 is not None:
                try:
                    with torch.no_grad():
                        for j, (vx, vcond, _vmask) in enumerate(val_loader):
                            polar_imgs = vx[:, 0].cpu().numpy()
                            _, vcont = vcond
                            for k in range(polar_imgs.shape[0]):
                                r_valid_rel_k = float(vcont[k, 1].item())
                                # Save polar image directly.
                                polar_u8 = ((polar_imgs[k] + 1.0) * 0.5 * 255.0).clip(0, 255).astype("uint8")
                                cv2.imwrite(
                                    os.path.join(val_pictures_dir, f"ep{epoch:04d}_val{j:02d}_{k:02d}_polar.png"),
                                    polar_u8,
                                )
                                # Save back-projected cartesian image.
                                cart_u8 = _polar_sample_to_uint8_cart(
                                    torch.from_numpy(polar_imgs[k]), r_valid_rel_k,
                                    cart_size, N_r, N_theta, pad_value,
                                )
                                cv2.imwrite(
                                    os.path.join(val_pictures_dir, f"ep{epoch:04d}_val{j:02d}_{k:02d}_cart.png"),
                                    cart_u8,
                                )
                except Exception:
                    pass

            # Qualitative sampling.
            if int(args.sample_every) > 0 and (epoch % int(args.sample_every) == 0):
                if cv2 is None:
                    warnings.warn("cv2 not available; skipping qualitative sampling.")
                else:
                    samples_dir = os.path.join(run_dir, "samples_training", f"epoch_{epoch:04d}")
                    os.makedirs(samples_dir, exist_ok=True)
                    try:
                        ema_model.eval()
                        with torch.no_grad():
                            cond_entries = []
                            for ci, (vx, vcond, vmask) in enumerate(val_loader):
                                cat_v, cont_v = vcond
                                cat_v = cat_v.to(device)
                                cont_v = cont_v.to(device)
                                cond_fixed = (
                                    cat_v[:1].expand(int(args.sample_n), -1),
                                    cont_v[:1].expand(int(args.sample_n), -1),
                                )
                                cond_entries.append({
                                    "cond_index": ci,
                                    "cat_ids": cat_v[:1].cpu().tolist()[0],
                                    "cont": cont_v[:1].cpu().tolist()[0],
                                })
                                r_valid_rel_cond = float(cont_v[0, 1].item())
                                r_valid_row = int(round(r_valid_rel_cond * (N_r - 1)))

                                # Use the actual mask from the val batch (from training data).
                                # vmask: [B, N_r, N_theta] → take first sample, expand to [sample_n, 1, N_r, N_theta]
                                mask_cond = vmask[:1, None].to(device=device, dtype=torch.float32)
                                mask_cond = mask_cond.expand(int(args.sample_n), -1, -1, -1).contiguous()

                                _sample_fn = _sample_ddim if args.sample_sampler == "ddim" else _sample_ddpm
                                _sample_kwargs = dict(
                                    model=ema_model, diffusion=diffusion, cond=cond_fixed,
                                    n=int(args.sample_n), N_r=N_r, N_theta=N_theta,
                                    r_valid_row=r_valid_row, pad_value=pad_value,
                                    cfg_scale=float(args.sample_cfg_scale), device=device,
                                    mask=mask_cond,
                                )
                                if args.sample_sampler == "ddim":
                                    _sample_kwargs["ddim_steps"] = int(args.sample_ddim_steps)
                                    _sample_kwargs["ddim_eta"]   = float(args.sample_ddim_eta)
                                gen = _sample_fn(**_sample_kwargs)
                                for si in range(gen.shape[0]):
                                    # Save polar sample.
                                    gen_u8 = _to_uint8_np(gen[si, 0].cpu().numpy())
                                    polar_path = os.path.join(samples_dir, f"cond{ci:02d}_s{si:02d}_polar.png")
                                    cv2.imwrite(polar_path, gen_u8)

                                    # Save back-projected cartesian sample.
                                    cart_u8 = _polar_sample_to_uint8_cart(
                                        gen[si, 0], r_valid_rel_cond,
                                        cart_size, N_r, N_theta, pad_value,
                                    )
                                    cart_path = os.path.join(samples_dir, f"cond{ci:02d}_s{si:02d}_cart.png")
                                    cv2.imwrite(cart_path, cart_u8)

                                    if clearml_logger is not None:
                                        clearml_logger.report_image(
                                            "samples_polar", f"epoch_{epoch:04d}",
                                            iteration=epoch, local_path=polar_path,
                                        )
                                        clearml_logger.report_image(
                                            "samples_cart", f"epoch_{epoch:04d}",
                                            iteration=epoch, local_path=cart_path,
                                        )

                            with open(os.path.join(samples_dir, "metadata.json"), "w", encoding="utf-8") as f:
                                json.dump({
                                    "epoch": epoch,
                                    "sample_every": int(args.sample_every),
                                    "sample_n": int(args.sample_n),
                                    "sample_cfg_scale": float(args.sample_cfg_scale),
                                    "N_r": N_r, "N_theta": N_theta,
                                    "val_selected_samples": val_samples,
                                    "sampling_conditions": cond_entries,
                                }, f, indent=2)
                    except Exception as e:
                        import traceback
                        print(f"[WARNING] Qualitative sampling failed: {e}")
                        traceback.print_exc()

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "N_r": N_r, "N_theta": N_theta,
                }, os.path.join(run_dir, "checkpoint_best.pt"))

            _append_csv_row(
                epoch_loss_csv,
                header=["epoch", "train_loss", "val_loss", "best_val"],
                row=[epoch, float(train_loss), float(val_loss), float(best_val)],
            )
            if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "N_r": N_r, "N_theta": N_theta,
                }, os.path.join(weights_dir, f"checkpoint_epoch_{epoch:04d}.pt"))

    # --- Picture-budget training loop ---
    else:
        pictures_seen = 0
        next_val_at = int(args.val_every_pictures) if int(args.val_every_pictures) > 0 else None
        last_val_loss: Optional[float] = None

        use_tqdm = (bool(args.tqdm) or sys.stderr.isatty()) and tqdm is not None
        train_iter = train_loader
        pbar = None
        if use_tqdm:
            pbar = tqdm(train_loader, desc="Training (max_pictures)", unit="batch", leave=False)
            train_iter = pbar

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for batch_idx, (x, cond, mask) in enumerate(train_iter, start=1):
            x = x.to(device)
            mask = mask.to(device)
            cat, cont = cond
            cat = cat.to(device)
            cont = cont.to(device)

            cond_in = None if (float(args.p_uncond) > 0 and random.random() < float(args.p_uncond)) else (cat, cont)

            t = diffusion.sample_timesteps(x.shape[0], device=device)
            x_t, noise = diffusion.noise_images(x, t)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(x_t, t, cond_in)
                loss_raw = masked_mse(pred, noise, mask)
                loss = loss_raw / max(1, int(args.accumulation_steps))

            scaler.scale(loss).backward()

            if batch_idx % max(1, int(args.accumulation_steps)) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.step_ema(ema_model, model, step_start_ema=0)

            running += float(loss.item())
            pictures_seen += int(x.shape[0])

            _append_csv_row(
                batch_loss_csv,
                header=["pictures_seen", "batch", "train_loss"],
                row=[pictures_seen, batch_idx, float(loss_raw.detach().item())],
            )
            if clearml_logger is not None:
                clearml_logger.report_scalar("loss", "train", iteration=pictures_seen, value=float(loss_raw.detach().item()))
            if pbar is not None:
                pbar.set_postfix({"loss": f"{float(loss.item()):.4f}", "pics": pictures_seen})

            if next_val_at is not None and pictures_seen >= int(next_val_at):
                while next_val_at is not None and pictures_seen >= int(next_val_at):
                    val_loss = run_eval(val_loader, model, ema_model, diffusion, device, use_ema=(not bool(args.no_ema_val)))
                    last_val_loss = float(val_loss)
                    if val_loss < best_val:
                        best_val = float(val_loss)
                        torch.save({
                            "epoch": 0, "pictures_seen": pictures_seen,
                            "model": model.state_dict(), "ema_model": ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(), "best_val": best_val,
                            "N_r": N_r, "N_theta": N_theta,
                        }, os.path.join(run_dir, "checkpoint_best.pt"))
                    _append_csv_row(
                        val_loss_csv,
                        header=["pictures_seen", "val_loss", "best_val", "use_ema_val"],
                        row=[pictures_seen, float(val_loss), float(best_val), (not bool(args.no_ema_val))],
                    )
                    if clearml_logger is not None:
                        clearml_logger.report_scalar("loss", "val", iteration=pictures_seen, value=float(val_loss))
                    next_val_at = int(next_val_at) + int(args.val_every_pictures)

        if next_val_at is None:
            val_loss = run_eval(val_loader, model, ema_model, diffusion, device, use_ema=(not bool(args.no_ema_val)))
            last_val_loss = float(val_loss)
            if val_loss < best_val:
                best_val = float(val_loss)
                torch.save({
                    "epoch": 0, "pictures_seen": pictures_seen,
                    "model": model.state_dict(), "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_val": best_val,
                    "N_r": N_r, "N_theta": N_theta,
                }, os.path.join(run_dir, "checkpoint_best.pt"))
            _append_csv_row(
                val_loss_csv,
                header=["pictures_seen", "val_loss", "best_val", "use_ema_val"],
                row=[pictures_seen, float(val_loss), float(best_val), (not bool(args.no_ema_val))],
            )

        train_loss = running / max(1, len(train_loader))
        val_loss_summary = float(last_val_loss) if last_val_loss is not None else float(best_val)
        _append_csv_row(
            epoch_loss_csv,
            header=["epoch", "train_loss", "val_loss", "best_val"],
            row=[0, float(train_loss), float(val_loss_summary), float(best_val)],
        )
        print(f"Done (max_pictures): pics={pictures_seen} | train={train_loss:.6f} | best_val={best_val:.6f}")

    # --- Final test evaluation ---
    use_ema_val = not bool(args.no_ema_val)
    test_loss = run_eval(test_loader, model, ema_model, diffusion, device, use_ema=use_ema_val)
    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "test_loss": test_loss,
            "best_val": best_val,
            "use_ema_val": bool(use_ema_val),
            "N_r": N_r, "N_theta": N_theta,
        }, f, indent=2)

    print(f"Test loss ({'ema' if use_ema_val else 'raw'}): {test_loss:.6f}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
