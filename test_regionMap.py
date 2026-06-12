"""Visual test for a region-weight map based on per-angle Mandrel/Ring detection.

This is a variant of the region-weight map used in masked_mse during polar
training (see CT_scan_model/dataset_ct_polar.py::_compute_region_weights),
but instead of a single, radially-symmetric Mandrel row it reuses the
per-angle boundary detection from test_region_detection.py
(detect_regions_per_angle), which returns

  r_mandrel_per_angle[N_theta]  -- Mandrel/Layer transition, wiggles with theta
  r_ring_per_angle[N_theta]     -- Layer/Can transition,     wiggles with theta

Both boosts therefore follow the actual, non-circular transition curves
instead of a single horizontal line.

The resulting weight map is still a pure loss-weighting matrix [N_r, N_theta]
(NOT an input channel to the model):

  weights[i, j] >= 1  inside the cell  (boosted near Mandrel- and Ring-curve)
  weights[i, j] == 0  outside the cell (padding, ignored in the loss)

Usage
-----
  python test_regionMap.py              # file dialog
  python test_regionMap.py image.png    # explicit path

Figures
-------
Figure 1 - Polarbild + Gewichtskarte
  Left   : Polarbild (grau) mit per-Winkel Mandrel-/Ring-/Zellgrenze
  Middle : Gewichtskarte (region_weights), Grenzkurven eingezeichnet
  Right  : Binaere Maske zum Vergleich

Figure 2 - Rueckprojektion in kartesische Darstellung
  Left   : Original mit detektierten Konturen
  Middle : Gewichtskarte kartesisch rueckprojiziert
  Right  : Overlay: Original + Gewichtskarte

Figure 3 - Profile
  Left   : Zeilenweises Profil der Gewichtskarte (Mandrel- vs. Ring-Boost)
  Right  : Histogramm der Gewichtswerte im gueltigen Bereich
"""

from __future__ import annotations

import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV (cv2) ist erforderlich: pip install opencv-python")

import config as project_config
from CT_scan_model.polar_transform import (
    detect_cell_boundary,
    cart_to_polar_boundary,
    polar_to_cart,
)
from test_region_detection import detect_regions_per_angle


# ---------------------------------------------------------------------------
# File picker
# ---------------------------------------------------------------------------

def _pick_file() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="CT-Scan Bild auswaehlen",
            filetypes=[("Image files", "*.png *.tif *.tiff *.bmp *.jpg *.jpeg"),
                       ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            raise SystemExit("Kein Bild ausgewaehlt.")
        return path
    except ImportError:
        raise SystemExit("tkinter nicht verfuegbar - Bildpfad als Argument uebergeben.")


# ---------------------------------------------------------------------------
# Region-Weight-Karte basierend auf per-Winkel Mandrel-/Ring-Grenzen
# ---------------------------------------------------------------------------

def compute_region_weights_per_angle(
    padding_mask: np.ndarray,
    r_mandrel_per_angle: np.ndarray,
    r_ring_per_angle: np.ndarray,
    w_mandrel: float = 3.0,
    w_ring: float = 8.0,
    sigma_mandrel: float = 12.0,
    sigma_ring: float = 6.0,
) -> np.ndarray:
    """Per-pixel loss weight map, both boosts following per-angle curves.

    Unlike CT_scan_model.dataset_ct_polar._compute_region_weights (where the
    Mandrel boost is a single radially-symmetric row), here BOTH the Mandrel
    boost and the Ring boost wiggle with theta, following
    r_mandrel_per_angle[theta] and r_ring_per_angle[theta] respectively
    (from test_region_detection.detect_regions_per_angle).

    This is the "smooth" (Gaussian-peak) variant -- only pixels close to the
    transition curves are upweighted, see
    compute_region_weights_piecewise() for a "blockwise constant" variant.

    Returns
    -------
    weights : float32 [N_r, N_theta], values >= 1 inside cell, 0 outside
    """
    N_r, N_theta = padding_mask.shape
    rows = np.arange(N_r, dtype=np.float32)[:, np.newaxis]   # [N_r, 1]

    mandrel_boost = (w_mandrel - 1.0) * np.exp(
        -((rows - r_mandrel_per_angle[np.newaxis, :]) / sigma_mandrel) ** 2
    )  # [N_r, N_theta]
    ring_boost = (w_ring - 1.0) * np.exp(
        -((rows - r_ring_per_angle[np.newaxis, :]) / sigma_ring) ** 2
    )  # [N_r, N_theta]

    w = 1.0 + mandrel_boost + ring_boost
    return (w * padding_mask).astype(np.float32)


def compute_region_weights_piecewise(
    padding_mask: np.ndarray,
    r_mandrel_per_angle: np.ndarray,
    r_ring_per_angle: np.ndarray,
    w_mandrel: float = 3.0,
    w_layers: float = 1.0,
    w_ring: float = 8.0,
) -> np.ndarray:
    """Blockwise-constant per-pixel loss weight map, 3 regions per angle.

    For each angular column theta, three radial regions (separated by the
    per-angle curves r_mandrel_per_angle[theta] and r_ring_per_angle[theta])
    each get ONE constant weight -- not just a narrow boost around the
    transition row, but the entire region:

      rows <  r_mandrel_per_angle[theta]                            -> w_mandrel
      r_mandrel_per_angle[theta] <= rows < r_ring_per_angle[theta]  -> w_layers
      rows >= r_ring_per_angle[theta]  (and inside the cell)        -> w_ring

    Outside the cell (padding_mask == 0) -> 0.

    Returns
    -------
    weights : float32 [N_r, N_theta]
    """
    N_r, N_theta = padding_mask.shape
    rows = np.arange(N_r, dtype=np.float32)[:, np.newaxis]   # [N_r, 1]

    is_mandrel = rows < r_mandrel_per_angle[np.newaxis, :]
    is_ring    = rows >= r_ring_per_angle[np.newaxis, :]
    is_layers  = ~is_mandrel & ~is_ring

    w = np.zeros((N_r, N_theta), dtype=np.float32)
    w[is_mandrel] = w_mandrel
    w[is_layers]  = w_layers
    w[is_ring]    = w_ring

    return (w * padding_mask).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        print("Oeffne Dateidialog...")
        img_path = _pick_file()

    if not os.path.isfile(img_path):
        raise SystemExit(f"Datei nicht gefunden: {img_path}")
    print(f"Bild: {img_path}")

    # --- Config ---
    N_r     = int(getattr(project_config, "POLAR_N_R",    512))
    N_theta = int(getattr(project_config, "POLAR_N_THETA", 1024))
    pad_val = float(getattr(project_config, "POLAR_PAD_VALUE", 0.0))

    # Region-weight hyperparameters (defaults from _compute_region_weights).
    w_mandrel = 3.0
    w_layers = 1.0
    w_ring = 8.0
    sigma_mandrel = 12.0
    sigma_ring = 6.0

    # "piecewise" -> blockwise-constant weights per region (Mandrel/Layers/Ring)
    # "smooth"    -> Gaussian peaks only around the transition curves
    weight_mode = "piecewise"

    # --- Bild laden (Originalgroesse) ---
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Bild konnte nicht geladen werden: {img_path}")
    H, W = gray.shape
    image_half_size = float(min(H, W)) / 2.0
    print(f"Originalgroesse: {W}x{H} px")

    # --- Zellgrenze detektieren ---
    cx, cy, r_valid_per_angle, edge_pts_xy, _ = detect_cell_boundary(
        gray, N_theta=N_theta, kernel_size=25, n_trace_angles=720,
    )
    r_max = float(np.max(r_valid_per_angle))
    print(f"Zellzentrum: cx={cx:.1f}, cy={cy:.1f}")
    print(f"r_max={r_max:.1f} px  r_valid_rel={r_max/image_half_size:.4f}")

    # --- Polartransformation ---
    img_norm = (gray.astype(np.float32) / 255.0) * 2.0 - 1.0
    polar_img, padding_mask, r_max_out = cart_to_polar_boundary(
        img_norm, cx, cy, r_valid_per_angle, N_r, N_theta, pad_val,
    )
    r_scale = r_max_out / max(1.0, float(N_r - 1))

    # --- Per-Winkel Mandrel-/Ring-Grenzen (test_region_detection.py) ---
    det = detect_regions_per_angle(
        polar_img, padding_mask, r_valid_per_angle,
        r_scale=r_scale, n_trace_angles=720,
        smooth_sigma=5.0, ring_search_frac=0.12,
        mandrel_search_frac=0.50,
    )
    r_mandrel_per_angle = det["r_mandrel_per_angle"]
    r_ring_per_angle    = det["r_ring_per_angle"]

    print(f"Mandrel-Grenze:  {r_mandrel_per_angle.mean():.1f} +/- {r_mandrel_per_angle.std():.1f} Zeilen")
    print(f"Ring-Grenze:     {r_ring_per_angle.mean():.1f} +/- {r_ring_per_angle.std():.1f} Zeilen")

    # --- Region-Weight-Karte (beide Grenzen per-Winkel) ---
    if weight_mode == "piecewise":
        region_weights = compute_region_weights_piecewise(
            padding_mask, r_mandrel_per_angle, r_ring_per_angle,
            w_mandrel=w_mandrel, w_layers=w_layers, w_ring=w_ring,
        )
    else:
        region_weights = compute_region_weights_per_angle(
            padding_mask, r_mandrel_per_angle, r_ring_per_angle,
            w_mandrel=w_mandrel, w_ring=w_ring,
            sigma_mandrel=sigma_mandrel, sigma_ring=sigma_ring,
        )

    valid_weights = region_weights[padding_mask > 0.5]
    print(f"Gewichtskarte: min={valid_weights.min():.4f}  "
          f"max={valid_weights.max():.4f}  mean={valid_weights.mean():.4f}  "
          f"Nullpixel (Padding): {(padding_mask < 0.5).sum()}")

    # Rueckprojektion der Gewichtskarte -> kartesisch
    weights_cart = polar_to_cart(region_weights, cx, cy, r_max_out, N_r, N_theta, W, pad_value=0.0)

    # Darstellungs-Arrays
    polar_u8 = np.clip((polar_img + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    theta_deg = np.linspace(0, 360, N_theta, endpoint=False)
    r_valid_rows = r_valid_per_angle / r_scale   # Zellgrenze im Polarbild

    vmax = max(w_mandrel, w_ring)
    ext = [0, 360, N_r, 0]   # imshow extent (origin='upper')

    # -------------------------------------------------------------------------
    # Figure 1 - Polarbild + Gewichtskarte + binaere Maske
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle("Region-Weight-Karte im Polarraum (per-Winkel Mandrel- & Ring-Grenze)", fontsize=13)

    axes1[0].imshow(polar_u8, cmap="gray", vmin=0, vmax=255,
                    aspect="auto", extent=ext)
    axes1[0].plot(theta_deg, r_mandrel_per_angle, color="dodgerblue", linewidth=1.5,
                  label="Mandrel-Grenze")
    axes1[0].plot(theta_deg, r_ring_per_angle, color="tomato", linewidth=1.5,
                  label="Ring-Grenze")
    axes1[0].plot(theta_deg, r_valid_rows, color="lime", linewidth=1.2,
                  linestyle="--", label="Zellgrenze (r_valid_per_angle)")
    axes1[0].set_title("Polarbild")
    axes1[0].set_xlabel("Winkel theta [Grad]")
    axes1[0].set_ylabel("Radius r [Zeilen-Index]")
    axes1[0].legend(fontsize=8)

    im1 = axes1[1].imshow(region_weights, cmap="inferno", vmin=0, vmax=vmax,
                           aspect="auto", extent=ext)
    axes1[1].plot(theta_deg, r_mandrel_per_angle, color="dodgerblue", linewidth=1.5,
                  linestyle="--", label="Mandrel-Boost folgt dieser Linie")
    axes1[1].plot(theta_deg, r_ring_per_angle, color="cyan", linewidth=1.5,
                  linestyle="--", label="Ring-Boost folgt dieser Linie")
    axes1[1].set_title("Region-Weight-Karte\n(1 = normal, hoeher = staerker gewichtet, 0 = Padding)")
    axes1[1].set_xlabel("Winkel theta [Grad]")
    axes1[1].set_ylabel("Radius r [Zeilen-Index]")
    axes1[1].legend(fontsize=8)
    plt.colorbar(im1, ax=axes1[1], fraction=0.03, pad=0.02, label="weight")

    axes1[2].imshow(padding_mask, cmap="gray", vmin=0, vmax=1,
                    aspect="auto", extent=ext)
    axes1[2].plot(theta_deg, r_valid_rows, color="lime", linewidth=1.5,
                  label="Zellgrenze")
    axes1[2].set_title("Binaere Maske\n(weiss = gueltig, schwarz = Padding)")
    axes1[2].set_xlabel("Winkel theta [Grad]")
    axes1[2].set_ylabel("Radius r [Zeilen-Index]")
    axes1[2].legend(fontsize=8)
    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 2 - Rueckprojektion kartesisch
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(19, 6))
    fig2.suptitle("Region-Weight-Karte - Rueckprojektion kartesisch", fontsize=13)

    axes2[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    if len(edge_pts_xy) >= 3:
        bx = np.append(edge_pts_xy[:, 0], edge_pts_xy[0, 0])
        by = np.append(edge_pts_xy[:, 1], edge_pts_xy[0, 1])
        axes2[0].plot(bx, by, color="lime", linewidth=1.5, linestyle="--", label="Zellgrenze")

    def _polar_curve_to_cart(r_per_angle):
        thetas = np.linspace(0.0, 2.0 * math.pi, N_theta, endpoint=False)
        r_px = r_per_angle * r_scale
        xs = cx + r_px * np.cos(thetas)
        ys = cy + r_px * np.sin(thetas)
        return np.append(xs, xs[0]), np.append(ys, ys[0])

    xs_m, ys_m = _polar_curve_to_cart(r_mandrel_per_angle)
    xs_r, ys_r = _polar_curve_to_cart(r_ring_per_angle)
    axes2[0].plot(xs_m, ys_m, color="dodgerblue", linewidth=1.5, label="Mandrel-Grenze")
    axes2[0].plot(xs_r, ys_r, color="tomato", linewidth=1.5, label="Ring-Grenze")
    axes2[0].legend(fontsize=8, loc="lower right")
    axes2[0].set_title(f"Original ({W}x{H})")
    axes2[0].axis("off")

    im2 = axes2[1].imshow(weights_cart, cmap="inferno", vmin=0, vmax=vmax)
    axes2[1].set_title("Gewichtskarte (kartesisch)\n0 = Padding -> hell = stark gewichtet")
    axes2[1].axis("off")
    plt.colorbar(im2, ax=axes2[1], fraction=0.03, pad=0.02, label="weight")

    # Overlay: Original + Gewichtskarte als Farbton
    gray_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    cmap_w = plt.get_cmap("inferno")
    w_norm = np.clip(weights_cart / vmax, 0, 1)
    w_rgba = cmap_w(w_norm)           # [H, W, 4]
    valid = (weights_cart > 1e-6)[..., np.newaxis]
    overlay = np.where(valid, 0.5 * gray_rgb + 0.5 * w_rgba[..., :3], gray_rgb)
    axes2[2].imshow(np.clip(overlay, 0, 1))
    axes2[2].set_title("Overlay: Original + Gewichtskarte")
    axes2[2].axis("off")
    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 3 - Profile / Histogramm
    # -------------------------------------------------------------------------
    row_mean = region_weights.mean(axis=1)
    row_min = region_weights.min(axis=1)
    row_max = region_weights.max(axis=1)

    fig3, (ax_prof, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("Region-Weight-Karte - Profile", fontsize=13)

    row_idx_axis = np.arange(N_r)
    ax_prof.plot(row_mean, row_idx_axis, color="#CC4422", linewidth=1.5, label="Mittelwert pro Zeile")
    ax_prof.fill_betweenx(row_idx_axis, row_min, row_max, color="#CC4422", alpha=0.2,
                          label="Min..Max ueber theta (Mandrel- & Ring-Wiggle)")
    ax_prof.axhline(N_r - 1, color="gray", linestyle=":", linewidth=1.0, label=f"Zeile {N_r-1} (r_max)")
    ax_prof.set_xlabel("Gewicht", fontsize=11)
    ax_prof.set_ylabel("Zeilen-Index", fontsize=11)
    ax_prof.set_title("Zeilenweises Gewichtsprofil\n(beide Boosts wiggeln jetzt mit theta)")
    ax_prof.invert_yaxis()
    ax_prof.legend(fontsize=9)
    ax_prof.grid(True, alpha=0.3)

    valid_vals = region_weights[padding_mask > 0.5].flatten()
    ax_hist.hist(valid_vals, bins=64, color="#CC4422", edgecolor="white", linewidth=0.3)
    ax_hist.set_xlabel("Gewichtswert", fontsize=11)
    ax_hist.set_ylabel("Pixel-Anzahl", fontsize=11)
    ax_hist.set_title("Histogramm der Gewichte (gueltiger Bereich)")
    ax_hist.grid(True, alpha=0.3)
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
