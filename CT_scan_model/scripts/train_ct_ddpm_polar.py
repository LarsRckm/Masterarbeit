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

import numpy as np
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

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")  # headless / cluster — no display
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

import math

from ..dataset_ct_polar import (
    BatteryCTPolarPerCellDataset,
    BatteryCTPolarSelectedSamplesDataset,
    BatteryCTPolarUniformCellsMaxPicturesDataset,
)
from ..diffusion_cartesian import Diffusion
from ..modules_cartesian_ct import UNet_conditional_cartesian
from ..polar_transform import polar_to_cart
from .sample_ct_ddpm_polar import (
    _sample_ddpm, _sample_ddim, _to_uint8_np, _polar_black_bg, DISPLAY_BG_VALUE,
)

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

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               region_weights: torch.Tensor = None) -> torch.Tensor:
    """Masked MSE loss, optionally weighted by per-pixel region weights."""
    if mask.dim() == 3:
        mask = mask[:, None, :, :]
    mask = mask.to(dtype=pred.dtype)
    if region_weights is not None:
        if region_weights.dim() == 3:
            region_weights = region_weights[:, None, :, :]
        w = mask * region_weights.to(dtype=pred.dtype)
    else:
        w = mask
    num = ((pred - target) ** 2 * w).sum()
    den = w.sum().clamp_min(1.0)
    return num / den


def region_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    region_labels: torch.Tensor,
    lam_mandrel: float = 1.0,
    lam_layers: float = 1.0,
    lam_housing: float = 1.0,
    lam_global: float = 0.0,
):
    """Per-region NORMALISED MSE on the prediction target (eps or v), summed with λ.

    ``region_labels``: [B, 1, H, W] or [B, H, W] with
        0 = padding, 1 = mandrel, 2 = layers, 3 = housing.

    Each region's MSE is the mean squared (pred - target) error over *that
    region's* pixels only — normalised by the region's pixel count, so its
    contribution is independent of how large the region is in a given image:

        L = λ_global · MSE_global
          + λ_mandrel · MSE_mandrel + λ_layers · MSE_layers + λ_housing · MSE_housing

    ``lam_global`` > 0 adds an area-weighted MSE over ALL valid (non-padding)
    pixels — the "hybrid" loss. The global term is area-coupled and therefore
    damps the global-brightness (DC) tug-of-war that the area-decoupled
    per-region terms create. With ``lam_global == 0`` only the per-region terms
    remain (pure per-region loss).

    Padding (label 0) is never included in any region nor the global term.

    Returns
    -------
    total      : scalar tensor (the loss to minimise)
    per_region : dict {"mandrel", "layers", "housing", "global"} of the
                 individual (un-weighted) MSEs, for logging.
    """
    if region_labels.dim() == 3:
        region_labels = region_labels[:, None, :, :]
    region_labels = region_labels.to(device=pred.device)

    # IMPORTANT: compute the loss in float32 even under autocast. The sums below
    # run over hundreds of thousands of pixels per region; in float16 (max ~65504)
    # both the squared errors and the reductions overflow, which corrupts the loss
    # value (and the logged per-region MSEs) into huge/garbage numbers.
    pred = pred.float()
    target = target.float()
    se = (pred - target) ** 2

    per_region = {}
    total = pred.new_zeros(())
    for cls, lam, name in ((1, lam_mandrel, "mandrel"),
                           (2, lam_layers, "layers"),
                           (3, lam_housing, "housing")):
        m = (region_labels == cls).to(dtype=torch.float32)
        denom = m.sum().clamp_min(1.0)
        mse_r = (se * m).sum() / denom
        per_region[name] = mse_r
        total = total + float(lam) * mse_r

    # Global (area-weighted, masked) term — hybrid loss when lam_global > 0.
    valid = (region_labels > 0).to(dtype=torch.float32)
    mse_global = (se * valid).sum() / valid.sum().clamp_min(1.0)
    per_region["global"] = mse_global
    if float(lam_global) > 0.0:
        total = total + float(lam_global) * mse_global

    return total, per_region


def gradient_loss(x0_pred: torch.Tensor, x0_true: torch.Tensor,
                  mask: torch.Tensor, alpha_hat_bchw: torch.Tensor) -> torch.Tensor:
    """ᾱ_t-weighted, masked MSE between the spatial gradients of x0_pred and x0_true.

    Finite differences in r (rows) and θ (cols). The adjacency mask counts a
    gradient only where BOTH neighbouring pixels are valid → the cell↔padding
    boundary is excluded automatically. ``alpha_hat_bchw`` ([B,1,1,1]) weights
    each sample by its signal fraction ᾱ_t (≈0 at high noise, where x0_pred is
    unreliable; ≈1 at low noise).
    """
    x0_pred = x0_pred.float(); x0_true = x0_true.float()
    mask = mask.float(); a = alpha_hat_bchw.float()

    drp = x0_pred[:, :, 1:, :] - x0_pred[:, :, :-1, :]
    drt = x0_true[:, :, 1:, :] - x0_true[:, :, :-1, :]
    mr  = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    dcp = x0_pred[:, :, :, 1:] - x0_pred[:, :, :, :-1]
    dct = x0_true[:, :, :, 1:] - x0_true[:, :, :, :-1]
    mc  = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    num = (a * mr * (drp - drt) ** 2).sum() + (a * mc * (dcp - dct) ** 2).sum()
    den = (mr.sum() + mc.sum()).clamp_min(1.0)
    return num / den


def grad_loss_term(pred: torch.Tensor, x: torch.Tensor, x_t: torch.Tensor,
                   t: torch.Tensor, labels: torch.Tensor, diffusion: Diffusion,
                   prediction_type: str, lam_grad: float) -> torch.Tensor:
    """λ_grad · ᾱ_t · gradient_loss(x̂0, x0), masked to the cell. 0 if λ_grad<=0.

    Reconstructs x̂0 from the model output (v or eps), compares its spatial
    gradients to the true x0's, weighted by ᾱ_t over t (sharpness is learned at
    low noise, where x̂0 is meaningful).
    """
    if float(lam_grad) <= 0.0:
        return pred.new_zeros(())
    x_t_img = x_t[:, :1]
    x0_true = x[:, :1]
    if prediction_type == "v":
        x0_pred = diffusion.v_to_x0(x_t_img.float(), pred.float(), t)
    else:
        a_sqrt = torch.sqrt(diffusion.alpha_hat[t])[:, None, None, None].float()
        s = torch.sqrt(1.0 - diffusion.alpha_hat[t])[:, None, None, None].float()
        x0_pred = (x_t_img.float() - s * pred.float()) / a_sqrt.clamp_min(1e-8)
    a = diffusion.alpha_hat[t][:, None, None, None]
    m = labels if labels.dim() == 4 else labels[:, None]
    m = (m > 0)
    return float(lam_grad) * gradient_loss(x0_pred, x0_true, m, a)


def _save_grad_plot(path: str, x_t: np.ndarray, x0: np.ndarray, x0_hat: np.ndarray,
                    mask: np.ndarray, t_val: int) -> None:
    """Diagnostic figure for the gradient loss (polar space, one sample):
      row 1 (spanning both cols): noised input x_t
      row 2: x0 (true)        | x0_hat (reconstructed)
      row 3: |grad x0|        | |grad x0_hat|
    """
    def _grad_mag(img: np.ndarray) -> np.ndarray:
        dr = np.zeros_like(img); dc = np.zeros_like(img)
        dr[:-1, :] = img[1:, :] - img[:-1, :]
        dc[:, :-1] = img[:, 1:] - img[:, :-1]
        return np.sqrt(dr * dr + dc * dc)

    m = (mask > 0.5).astype(np.float32)
    g0 = _grad_mag(x0) * m
    gh = _grad_mag(x0_hat) * m
    gmax = float(max(g0.max(), gh.max(), 1e-6))

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(3, 2)
    ax = fig.add_subplot(gs[0, :]); ax.imshow(x_t, cmap="gray", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"x_t  (noised input, t={t_val})"); ax.axis("off")
    ax = fig.add_subplot(gs[1, 0]); ax.imshow(x0, cmap="gray", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("x0  (true)"); ax.axis("off")
    ax = fig.add_subplot(gs[1, 1]); ax.imshow(x0_hat, cmap="gray", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("x0_hat  (reconstructed)"); ax.axis("off")
    ax = fig.add_subplot(gs[2, 0]); ax.imshow(g0, cmap="inferno", vmin=0, vmax=gmax, aspect="auto")
    ax.set_title("|grad x0|"); ax.axis("off")
    ax = fig.add_subplot(gs[2, 1]); ax.imshow(gh, cmap="inferno", vmin=0, vmax=gmax, aspect="auto")
    ax.set_title("|grad x0_hat|"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


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
    lam_mandrel: float = 1.0,
    lam_layers: float = 1.0,
    lam_housing: float = 1.0,
    lam_global: float = 0.0,
    prediction_type: str = "eps",
    offset_noise: float = 0.0,
    lam_grad: float = 0.0,
):
    """Returns (total_loss, {"mandrel","layers","housing","global"}) — all means."""
    net = ema_model if use_ema else model
    net.eval()
    total = 0.0
    count = 0
    reg_acc = {"mandrel": 0.0, "layers": 0.0, "housing": 0.0, "global": 0.0}
    with torch.no_grad():
        for batch in loader:
            x, cond = batch[0], batch[1]
            labels = batch[3] if len(batch) > 3 else None
            bs = int(x.shape[0])
            x = x.to(device)
            if labels is not None:
                labels = labels.to(device)
            cat, cont = cond
            cat = cat.to(device)
            cont = cont.to(device)
            t = diffusion.sample_timesteps(bs, device=device)
            x_t, noise = diffusion.noise_images(x, t, offset_noise=offset_noise)
            pred = net(x_t, t, (cat, cont))
            target = diffusion.get_v(x[:, :1], noise, t) if prediction_type == "v" else noise
            loss_t, per_region = region_mse_loss(
                pred, target, labels, lam_mandrel, lam_layers, lam_housing, lam_global,
            )
            loss_t = loss_t + grad_loss_term(
                pred, x, x_t, t, labels, diffusion, prediction_type, lam_grad)
            total += float(loss_t.item()) * bs
            for k in reg_acc:
                reg_acc[k] += float(per_region[k].item()) * bs
            count += bs
    net.train()
    n = max(1, count)
    return float(total / n), {k: reg_acc[k] / n for k in reg_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Train polar CT DDPM (single GPU).")
    p.add_argument("--index", default=os.path.join("CT_scan_model", "cell_index.json"))
    p.add_argument("--geometry", default=os.path.join("CT_scan_model", "cell_geometry.json"))
    p.add_argument("--splits", default=os.path.join("CT_scan_model", "splits.json"))

    p.add_argument("--slices-per-cell", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--max-pictures", type=int, default=0)
    p.add_argument("--val-every-pictures", type=int, default=0)
    p.add_argument("--no-ema-val", action="store_true")

    p.add_argument("--noise-steps", type=int, default=1000)
    p.add_argument("--beta-schedule", choices=["linear", "cosine"], default="linear",
                   help="Noise schedule. Default 'linear' (V6). 'cosine' spends more steps in the "
                        "low-noise regime but makes the DDPM reverse step unstable at high t "
                        "(1/sqrt(alpha) blows up), which destabilised V7.")
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--ema-beta", type=float, default=0.999,
                   help="EMA decay (V6 value 0.995). Higher = smoother/more stable EMA model "
                        "for the qualitative samples.")
    p.add_argument("--p-uncond", type=float, default=0.1)

    p.add_argument("--lambda-mandrel", type=float, default=1.0,
                   help="λ for the per-region-normalised Mandrel MSE term.")
    p.add_argument("--lambda-schichten", type=float, default=1.0,
                   help="λ for the per-region-normalised layers (winding) MSE term.")
    p.add_argument("--lambda-gehaeuse", type=float, default=1.0,
                   help="λ for the per-region-normalised housing/can MSE term.")

    p.add_argument("--prediction-type", choices=["eps", "v"], default="v",
                   help="Model target: 'eps' (noise) or 'v' (velocity, Salimans & Ho). "
                        "v makes the high-t target contain x0 -> better global brightness. "
                        "Must match between training and sampling.")
    p.add_argument("--zero-terminal-snr", action="store_true",
                   help="Rescale the schedule so alpha_hat[-1]=0 (Lin et al.) -> the terminal "
                        "step is pure noise, removing the brightness train/test mismatch. "
                        "Requires --prediction-type v. Must match between training and sampling.")
    p.add_argument("--loss-type", choices=["global", "region"], default="global",
                   help="'region': only the per-region terms (λ_m/λ_l/λ_h). "
                        "'global': HYBRID = an area-weighted global MSE term (λ_global) PLUS "
                        "the per-region terms — the global term damps the brightness oscillation.")
    p.add_argument("--lambda-global", type=float, default=1.0,
                   help="λ for the area-weighted global MSE term (only used with --loss-type global).")
    p.add_argument("--offset-noise", type=float, default=0.0,
                   help="Offset-noise strength c (0 = off, ~0.1 = on). Adds a per-image DC offset "
                        "to the training noise so the model learns the global brightness.")
    p.add_argument("--lambda-grad", type=float, default=1.0,
                   help="Max weight of the gradient (edge-sharpness) loss (0 = off). Penalises "
                        "blurred spatial gradients of the reconstructed x0 vs the true x0, "
                        "weighted by ᾱ_t over t (acts mainly at low noise). Fixes oversmoothed "
                        "thin/thick layers, stripe edges and tab corners.")

    p.add_argument("--no-radial-map", action="store_true",
                   help="A/B ablation: zero the radial-map conditioning channel (ch2). "
                        "Functionally equivalent to a 2-channel model (image+mask). Must be "
                        "set identically for training and sampling of the same checkpoint.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--tqdm", action="store_true", default=True)

    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Stop training (epoch-based mode) if val_loss does not improve "
                        "for this many consecutive epochs. Set to 0 to disable.")

    p.add_argument("--resume", action="store_true",
                   help="Resume training from a checkpoint in --run-dir")
    p.add_argument("--resume-from", default=None,
                   help="Explicit checkpoint path to resume from (overrides default search)")

    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--sample-n", type=int, default=1)
    p.add_argument("--val-pictures-every", type=int, default=0,
                   help="Save the ground-truth val_pictures every N epochs (0 = disable). "
                        "These are static reference images; >1 avoids redundant re-saving.")
    p.add_argument("--grad-plot-every", type=int, default=5,
                   help="Save a gradient-loss diagnostic figure (x_t, x0, x0_hat and their "
                        "gradients) every N epochs (0 = disable). Requires matplotlib.")
    p.add_argument("--sample-cfg-scale", type=float, default=1.0)
    p.add_argument("--sample-sampler", choices=["ddpm", "ddim"], default="ddpm",
                   help="Sampler used for qualitative samples during training")
    p.add_argument("--sample-ddim-steps", type=int, default=200,
                   help="DDIM steps (only when --sample-sampler=ddim)")
    p.add_argument("--sample-ddim-eta", type=float, default=0.0,
                   help="DDIM eta: 0=deterministic, 1=full noise")

    p.add_argument("--no-clearml", action="store_true", default=True)
    args = p.parse_args(argv)

    # Per-region loss weights (λ) — reused for training, validation and test.
    lam_m = float(args.lambda_mandrel)
    lam_l = float(args.lambda_schichten)
    lam_h = float(args.lambda_gehaeuse)
    # Global (area-weighted) term only active in the hybrid 'global' loss type.
    lam_g = float(args.lambda_global) if str(args.loss_type) == "global" else 0.0
    pred_type = str(args.prediction_type)
    offset_noise = float(args.offset_noise)
    lam_grad = float(args.lambda_grad)
    use_radial = not bool(args.no_radial_map)   # radial-map A/B ablation
    if bool(args.zero_terminal_snr) and pred_type != "v":
        raise SystemExit("--zero-terminal-snr requires --prediction-type v "
                         "(eps-prediction is degenerate at alpha_hat=0).")

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
            use_radial_map=use_radial,
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
            use_radial_map=use_radial,
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

    val_ds = BatteryCTPolarSelectedSamplesDataset(
        args.index, args.geometry, samples=val_samples, use_radial_map=use_radial)
    test_ds = BatteryCTPolarSelectedSamplesDataset(
        args.index, args.geometry, samples=test_samples, use_radial_map=use_radial)

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
        schedule=str(args.beta_schedule),
        zero_terminal_snr=bool(args.zero_terminal_snr),
    ).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    start_epoch = 1
    epochs_no_improve = 0
    prev_recent_path = None   # rolling --save-every 0 checkpoint (checkpoint_{epoch}.pt)

    # --- Resume from checkpoint ---
    if bool(args.resume) or args.resume_from is not None:
        if args.resume_from is not None:
            ckpt_path = args.resume_from
        else:
            # Priority: rolling checkpoint_{epoch}.pt with the highest epoch
            #           (save-every=0) → checkpoint_best.pt → latest epoch checkpoint
            import glob as _glob
            import re as _re
            most_recent = None
            best_ep = -1
            for c in _glob.glob(os.path.join(run_dir, "checkpoint_*.pt")):
                m = _re.match(r"checkpoint_(\d+)\.pt$", os.path.basename(c))
                if m and int(m.group(1)) > best_ep:
                    best_ep = int(m.group(1))
                    most_recent = c
            best_ckpt = os.path.join(run_dir, "checkpoint_best.pt")
            if most_recent is not None:
                ckpt_path = most_recent
            elif os.path.isfile(best_ckpt):
                ckpt_path = best_ckpt
            else:
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
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"  → continuing from epoch {start_epoch}  (best_val={best_val:.6f}, "
              f"epochs_no_improve={epochs_no_improve})")

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
            running_region = {"mandrel": 0.0, "layers": 0.0, "housing": 0.0, "global": 0.0}

            use_tqdm = (bool(args.tqdm) or sys.stderr.isatty()) and tqdm is not None
            train_iter = train_loader
            pbar = None
            if use_tqdm:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{int(args.epochs)}", unit="batch", leave=False)
                train_iter = pbar

            for step, batch in enumerate(train_iter, start=1):
                x, cond = batch[0], batch[1]
                labels = batch[3] if len(batch) > 3 else None
                global_step += 1
                x = x.to(device)
                if labels is not None:
                    labels = labels.to(device)
                cat, cont = cond
                cat = cat.to(device)
                cont = cont.to(device)

                cond_in = None if (float(args.p_uncond) > 0 and random.random() < float(args.p_uncond)) else (cat, cont)

                t = diffusion.sample_timesteps(x.shape[0], device=device)
                x_t, noise = diffusion.noise_images(x, t, offset_noise=offset_noise)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred = model(x_t, t, cond_in)
                    target = diffusion.get_v(x[:, :1], noise, t) if pred_type == "v" else noise
                    loss_raw, region_mse = region_mse_loss(
                        pred, target, labels, lam_m, lam_l, lam_h, lam_g)
                    loss_raw = loss_raw + grad_loss_term(
                        pred, x, x_t, t, labels, diffusion, pred_type, lam_grad)
                    loss = loss_raw / max(1, int(args.accumulation_steps))

                scaler.scale(loss).backward()

                if step % max(1, int(args.accumulation_steps)) == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.step_ema(ema_model, model, step_start_ema=0)

                running += float(loss.item())
                for k in running_region:
                    running_region[k] += float(region_mse[k].detach().item())
                _append_csv_row(
                    batch_loss_csv,
                    header=["epoch", "batch", "train_loss",
                            "mse_mandrel", "mse_layers", "mse_housing", "mse_global"],
                    row=[epoch, step, float(loss_raw.detach().item()),
                         float(region_mse["mandrel"].detach().item()),
                         float(region_mse["layers"].detach().item()),
                         float(region_mse["housing"].detach().item()),
                         float(region_mse["global"].detach().item())],
                )
                if clearml_logger is not None:
                    clearml_logger.report_scalar("loss", "train", iteration=global_step, value=float(loss_raw.detach().item()))
                if pbar is not None:
                    pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

            nb = max(1, len(train_loader))
            train_loss = running / nb
            train_region = {k: running_region[k] / nb for k in running_region}
            val_loss, val_region = run_eval(
                val_loader, model, ema_model, diffusion, device,
                use_ema=(not bool(args.no_ema_val)),
                lam_mandrel=lam_m, lam_layers=lam_l, lam_housing=lam_h, lam_global=lam_g,
                prediction_type=pred_type, offset_noise=offset_noise, lam_grad=lam_grad,
            )
            print(f"Epoch {epoch:04d} | train={train_loss:.6f} | "
                  f"val={'ema' if not args.no_ema_val else 'raw'}={val_loss:.6f} | "
                  f"val MSE M/L/H={val_region['mandrel']:.4f}/{val_region['layers']:.4f}/{val_region['housing']:.4f}")

            if clearml_logger is not None:
                clearml_logger.report_scalar("loss_epoch", "train", iteration=epoch, value=float(train_loss))
                clearml_logger.report_scalar("loss_epoch", "val", iteration=epoch, value=float(val_loss))

            # Save ground-truth polar/cartesian reference images (black background).
            if (cv2 is not None and int(args.val_pictures_every) > 0
                    and (epoch % int(args.val_pictures_every) == 0)):
                try:
                    with torch.no_grad():
                        for j, val_batch in enumerate(val_loader):
                            vx, vcond = val_batch[0], val_batch[1]
                            polar_imgs = vx[:, 0].cpu().numpy()
                            mask_imgs  = vx[:, 1].cpu().numpy()
                            _, vcont = vcond
                            for k in range(polar_imgs.shape[0]):
                                r_valid_rel_k = float(vcont[k, 1].item())
                                polar_disp = _polar_black_bg(polar_imgs[k], mask_imgs[k])
                                # Save polar image (black background).
                                cv2.imwrite(
                                    os.path.join(val_pictures_dir, f"ep{epoch:04d}_val{j:02d}_{k:02d}_polar.png"),
                                    _to_uint8_np(polar_disp),
                                )
                                # Save back-projected cartesian image (black background).
                                cart_u8 = _polar_sample_to_uint8_cart(
                                    torch.from_numpy(polar_disp), r_valid_rel_k,
                                    cart_size, N_r, N_theta, DISPLAY_BG_VALUE,
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
                            for ci, val_batch in enumerate(val_loader):
                                vx, vcond, vmask = val_batch[0], val_batch[1], val_batch[2]
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

                                # Channel 1 = binary mask, channel 2 = radial map — from val batch
                                mask_cond   = vx[:1, 1:2].to(device=device, dtype=torch.float32)
                                radial_cond = vx[:1, 2:3].to(device=device, dtype=torch.float32)
                                mask_cond   = mask_cond.expand(int(args.sample_n),   -1, -1, -1).contiguous()
                                radial_cond = radial_cond.expand(int(args.sample_n), -1, -1, -1).contiguous()

                                _sample_fn = _sample_ddim if args.sample_sampler == "ddim" else _sample_ddpm
                                _sample_kwargs = dict(
                                    model=ema_model, diffusion=diffusion, cond=cond_fixed,
                                    n=int(args.sample_n), N_r=N_r, N_theta=N_theta,
                                    r_valid_row=r_valid_row, pad_value=pad_value,
                                    cfg_scale=float(args.sample_cfg_scale), device=device,
                                    mask=mask_cond, radial_map=radial_cond,
                                    prediction_type=pred_type,
                                )
                                if args.sample_sampler == "ddim":
                                    _sample_kwargs["ddim_steps"] = int(args.sample_ddim_steps)
                                    _sample_kwargs["ddim_eta"]   = float(args.sample_ddim_eta)
                                gen = _sample_fn(**_sample_kwargs)
                                for si in range(gen.shape[0]):
                                    # Render padding as black (display only).
                                    polar_disp = _polar_black_bg(
                                        gen[si, 0].cpu().numpy(),
                                        mask_cond[si, 0].cpu().numpy(),
                                    )
                                    # Save polar sample.
                                    gen_u8 = _to_uint8_np(polar_disp)
                                    polar_path = os.path.join(samples_dir, f"cond{ci:02d}_s{si:02d}_polar.png")
                                    cv2.imwrite(polar_path, gen_u8)

                                    # Save back-projected cartesian sample (black background).
                                    cart_u8 = _polar_sample_to_uint8_cart(
                                        torch.from_numpy(polar_disp), r_valid_rel_cond,
                                        cart_size, N_r, N_theta, DISPLAY_BG_VALUE,
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

            # Gradient-loss diagnostic figure (x_t, x0, x0_hat + their gradients).
            if (HAS_MPL and int(args.grad_plot_every) > 0
                    and (epoch % int(args.grad_plot_every) == 0)):
                try:
                    grad_dir = os.path.join(run_dir, "grad_plots")
                    os.makedirs(grad_dir, exist_ok=True)
                    t_plot = max(1, int(args.noise_steps) // 10)
                    with torch.no_grad():
                        vb = next(iter(val_loader))
                        vx = vb[0][:1].to(device)            # one sample [1, 3, H, W]
                        cat_v, cont_v = vb[1]
                        cond_v = (cat_v[:1].to(device), cont_v[:1].to(device))
                        t_fixed = torch.full((1,), t_plot, device=device, dtype=torch.long)
                        x_t, _ = diffusion.noise_images(vx, t_fixed, offset_noise=offset_noise)
                        pred = model(x_t, t_fixed, cond_v)
                        if pred_type == "v":
                            x0_hat = diffusion.v_to_x0(x_t[:, :1].float(), pred.float(), t_fixed)
                        else:
                            a_sqrt = torch.sqrt(diffusion.alpha_hat[t_fixed])[:, None, None, None].float()
                            s = torch.sqrt(1.0 - diffusion.alpha_hat[t_fixed])[:, None, None, None].float()
                            x0_hat = (x_t[:, :1].float() - s * pred.float()) / a_sqrt.clamp_min(1e-8)
                    _save_grad_plot(
                        os.path.join(grad_dir, f"ep{epoch:04d}_t{t_plot}.png"),
                        x_t[0, 0].cpu().numpy(),       # noised input
                        vx[0, 0].cpu().numpy(),        # x0 (true)
                        x0_hat[0, 0].cpu().numpy(),    # x0_hat
                        vx[0, 1].cpu().numpy(),        # mask channel
                        t_plot,
                    )
                except Exception as e:
                    import traceback
                    print(f"[WARNING] grad plot failed: {e}")
                    traceback.print_exc()

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "epochs_no_improve": epochs_no_improve,
                    "N_r": N_r, "N_theta": N_theta,
                }, os.path.join(run_dir, "checkpoint_best.pt"))
            else:
                epochs_no_improve += 1

            _append_csv_row(
                epoch_loss_csv,
                header=["epoch", "train_loss", "val_loss", "best_val",
                        "train_mse_mandrel", "train_mse_layers", "train_mse_housing", "train_mse_global",
                        "val_mse_mandrel", "val_mse_layers", "val_mse_housing", "val_mse_global"],
                row=[epoch, float(train_loss), float(val_loss), float(best_val),
                     float(train_region["mandrel"]), float(train_region["layers"]),
                     float(train_region["housing"]), float(train_region["global"]),
                     float(val_region["mandrel"]), float(val_region["layers"]),
                     float(val_region["housing"]), float(val_region["global"])],
            )
            ckpt_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "epochs_no_improve": epochs_no_improve,
                "N_r": N_r, "N_theta": N_theta,
            }
            if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
                torch.save(ckpt_state, os.path.join(weights_dir, f"checkpoint_epoch_{epoch:04d}.pt"))
            elif int(args.save_every) == 0:
                # --save-every 0: keep a single rolling checkpoint named by epoch
                # (so the epoch is readable from the filename), parallel to
                # checkpoint_best.pt. Save the new one first, then remove the
                # previous epoch's file so only the latest remains.
                recent_path = os.path.join(run_dir, f"checkpoint_{epoch}.pt")
                torch.save(ckpt_state, recent_path)
                if prev_recent_path is not None and prev_recent_path != recent_path:
                    try:
                        os.remove(prev_recent_path)
                    except OSError:
                        pass
                prev_recent_path = recent_path

            if int(args.early_stopping_patience) > 0 and epochs_no_improve >= int(args.early_stopping_patience):
                print(f"Early stopping: val_loss did not improve for {epochs_no_improve} epochs "
                      f"(patience={args.early_stopping_patience}). Stopping at epoch {epoch}.")
                break

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

        for batch_idx, batch in enumerate(train_iter, start=1):
            x, cond = batch[0], batch[1]
            labels = batch[3] if len(batch) > 3 else None
            x = x.to(device)
            if labels is not None:
                labels = labels.to(device)
            cat, cont = cond
            cat = cat.to(device)
            cont = cont.to(device)

            cond_in = None if (float(args.p_uncond) > 0 and random.random() < float(args.p_uncond)) else (cat, cont)

            t = diffusion.sample_timesteps(x.shape[0], device=device)
            x_t, noise = diffusion.noise_images(x, t, offset_noise=offset_noise)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(x_t, t, cond_in)
                target = diffusion.get_v(x[:, :1], noise, t) if pred_type == "v" else noise
                loss_raw, _ = region_mse_loss(
                    pred, target, labels, lam_m, lam_l, lam_h, lam_g)
                loss_raw = loss_raw + grad_loss_term(
                    pred, x, x_t, t, labels, diffusion, pred_type, lam_grad)
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
                    val_loss, _ = run_eval(val_loader, model, ema_model, diffusion, device,
                                   use_ema=(not bool(args.no_ema_val)),
                                   lam_mandrel=lam_m, lam_layers=lam_l, lam_housing=lam_h,
                                   lam_global=lam_g, prediction_type=pred_type, offset_noise=offset_noise,
                                   lam_grad=lam_grad)
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
            val_loss, _ = run_eval(val_loader, model, ema_model, diffusion, device,
                                   use_ema=(not bool(args.no_ema_val)),
                                   lam_mandrel=lam_m, lam_layers=lam_l, lam_housing=lam_h,
                                   lam_global=lam_g, prediction_type=pred_type, offset_noise=offset_noise,
                                   lam_grad=lam_grad)
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
    test_loss, test_region = run_eval(
        test_loader, model, ema_model, diffusion, device, use_ema=use_ema_val,
        lam_mandrel=lam_m, lam_layers=lam_l, lam_housing=lam_h, lam_global=lam_g,
        prediction_type=pred_type, offset_noise=offset_noise, lam_grad=lam_grad,
    )
    with open(os.path.join(run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "test_loss": test_loss,
            "test_mse_mandrel": test_region["mandrel"],
            "test_mse_layers": test_region["layers"],
            "test_mse_housing": test_region["housing"],
            "best_val": best_val,
            "use_ema_val": bool(use_ema_val),
            "lambda_mandrel": lam_m, "lambda_schichten": lam_l, "lambda_gehaeuse": lam_h,
            "N_r": N_r, "N_theta": N_theta,
        }, f, indent=2)

    print(f"Test loss ({'ema' if use_ema_val else 'raw'}): {test_loss:.6f} "
          f"| M/L/H={test_region['mandrel']:.4f}/{test_region['layers']:.4f}/{test_region['housing']:.4f}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
