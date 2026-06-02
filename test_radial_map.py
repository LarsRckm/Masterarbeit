"""Visual test for the radial position map (Channel 2 / c_in=3).

The radial map encodes the normalised radius per pixel:
  r_map[i, j] = (i / (N_r - 1))  inside  the cell boundary  → [0, 1]
  r_map[i, j] = 0                 outside the cell boundary

It is identical in shape to the binary padding mask but carries a
continuous radial gradient instead of a binary flag.

Usage
-----
  python test_radial_map.py              # file dialog
  python test_radial_map.py image.png    # explicit path

Figures
-------
Figure 1 — Polarbild + radiale Karte
  Left   : Polarbild (grau)
  Middle : Radiale Karte im Polarraum (Farbkodierung Viridis)
  Right  : Binäre Maske zum Vergleich

Figure 2 — Rückprojektion in kartesische Darstellung
  Left   : Original
  Middle : Radiale Karte kartesisch rückprojiziert
  Right  : Overlay: Original + radiale Karte

Figure 3 — Werteverteilung
  Left   : Histogramm der Werte im validen Bereich
  Right  : Zeilenweise Mittelwert der radialen Karte (Profil)
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
            title="CT-Scan Bild auswählen",
            filetypes=[("Image files", "*.png *.tif *.tiff *.bmp *.jpg *.jpeg"),
                       ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            raise SystemExit("Kein Bild ausgewählt.")
        return path
    except ImportError:
        raise SystemExit("tkinter nicht verfügbar — Bildpfad als Argument übergeben.")


# ---------------------------------------------------------------------------
# Radiale Karte berechnen
# ---------------------------------------------------------------------------

def build_radial_map(padding_mask: np.ndarray, N_r: int, N_theta: int) -> np.ndarray:
    """Radiale Positionskarte: r_map[i,j] = i/(N_r-1) innerhalb der Maske, 0 außerhalb.

    Parameters
    ----------
    padding_mask : float32 [N_r, N_theta] — 1 inside, 0 outside
    N_r, N_theta : Polarbild-Dimensionen

    Returns
    -------
    r_map : float32 [N_r, N_theta], Werte in [0, 1]
    """
    row_idx = np.arange(N_r, dtype=np.float32) / max(1.0, float(N_r - 1))  # [N_r]
    r_map   = row_idx[:, np.newaxis] * np.ones((1, N_theta), dtype=np.float32)
    r_map  *= padding_mask   # außerhalb der Zelle → 0
    return r_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        print("Öffne Dateidialog…")
        img_path = _pick_file()

    if not os.path.isfile(img_path):
        raise SystemExit(f"Datei nicht gefunden: {img_path}")
    print(f"Bild: {img_path}")

    # --- Config ---
    N_r     = int(getattr(project_config, "POLAR_N_R",    512))
    N_theta = int(getattr(project_config, "POLAR_N_THETA", 1024))
    pad_val = float(getattr(project_config, "POLAR_PAD_VALUE", 0.0))

    # --- Bild laden (Originalgröße) ---
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Bild konnte nicht geladen werden: {img_path}")
    H, W = gray.shape
    image_half_size = float(min(H, W)) / 2.0
    print(f"Originalgröße: {W}×{H} px")

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

    # --- Radiale Karte berechnen ---
    r_map = build_radial_map(padding_mask, N_r, N_theta)

    print(f"Radiale Karte: min={r_map[padding_mask > 0.5].min():.4f}  "
          f"max={r_map[padding_mask > 0.5].max():.4f}  "
          f"Nullpixel (Padding): {(padding_mask < 0.5).sum()}")

    # Rückprojektion der radialen Karte → kartesisch
    r_map_cart = polar_to_cart(r_map, cx, cy, r_max_out, N_r, N_theta, W, pad_value=0.0)

    # Darstellungs-Arrays
    polar_u8   = np.clip((polar_img + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    theta_deg  = np.linspace(0, 360, N_theta, endpoint=False)
    r_scale    = r_max_out / max(1.0, float(N_r - 1))
    r_valid_rows = r_valid_per_angle / r_scale   # Grenzlinie im Polarbild

    ext = [0, 360, N_r, 0]   # imshow extent (origin='upper')

    # -------------------------------------------------------------------------
    # Figure 1 — Polarbild + radiale Karte + binäre Maske
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle("Radiale Positionskarte im Polarraum", fontsize=13)

    axes1[0].imshow(polar_u8, cmap="gray", vmin=0, vmax=255,
                    aspect="auto", extent=ext)
    axes1[0].plot(theta_deg, r_valid_rows, color="lime", linewidth=1.5,
                  label="Zellgrenze")
    axes1[0].set_title("Polarbild")
    axes1[0].set_xlabel("Winkel θ [°]")
    axes1[0].set_ylabel("Radius r [Zeilen-Index]")
    axes1[0].legend(fontsize=8)

    im1 = axes1[1].imshow(r_map, cmap="viridis", vmin=0, vmax=1,
                           aspect="auto", extent=ext)
    axes1[1].plot(theta_deg, r_valid_rows, color="red", linewidth=1.5,
                  linestyle="--", label="Zellgrenze")
    axes1[1].set_title("Radiale Karte\n(0 = Zentrum / Padding,  1 = äußerster Rand)")
    axes1[1].set_xlabel("Winkel θ [°]")
    axes1[1].set_ylabel("Radius r [Zeilen-Index]")
    axes1[1].legend(fontsize=8)
    plt.colorbar(im1, ax=axes1[1], fraction=0.03, pad=0.02, label="r_norm")

    axes1[2].imshow(padding_mask, cmap="gray", vmin=0, vmax=1,
                    aspect="auto", extent=ext)
    axes1[2].plot(theta_deg, r_valid_rows, color="lime", linewidth=1.5,
                  label="Zellgrenze")
    axes1[2].set_title("Binäre Maske\n(weiß = gültig,  schwarz = Padding)")
    axes1[2].set_xlabel("Winkel θ [°]")
    axes1[2].set_ylabel("Radius r [Zeilen-Index]")
    axes1[2].legend(fontsize=8)
    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 2 — Rückprojektion kartesisch
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(19, 6))
    fig2.suptitle("Radiale Karte — Rückprojektion kartesisch", fontsize=13)

    axes2[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    if len(edge_pts_xy) >= 3:
        bx = np.append(edge_pts_xy[:, 0], edge_pts_xy[0, 0])
        by = np.append(edge_pts_xy[:, 1], edge_pts_xy[0, 1])
        axes2[0].plot(bx, by, color="lime", linewidth=1.5)
    axes2[0].set_title(f"Original ({W}×{H})")
    axes2[0].axis("off")

    im2 = axes2[1].imshow(r_map_cart, cmap="viridis", vmin=0, vmax=1)
    axes2[1].set_title("Radiale Karte (kartesisch)\n0 = Zentrum/Außen  →  1 = Rand")
    axes2[1].axis("off")
    plt.colorbar(im2, ax=axes2[1], fraction=0.03, pad=0.02, label="r_norm")

    # Overlay: Original + radiale Karte als Farbton
    gray_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    cmap_v   = plt.get_cmap("viridis")
    r_rgba   = cmap_v(r_map_cart)           # [H, W, 4]
    valid    = (r_map_cart > 0.01)[..., np.newaxis]
    overlay  = np.where(valid, 0.5 * gray_rgb + 0.5 * r_rgba[..., :3], gray_rgb)
    axes2[2].imshow(np.clip(overlay, 0, 1))
    axes2[2].set_title("Overlay: Original + Radiale Karte")
    axes2[2].axis("off")
    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 3 — Werteverteilung
    # -------------------------------------------------------------------------
    valid_vals = r_map[padding_mask > 0.5].flatten()
    row_mean   = r_map.mean(axis=1)   # Mittelwert pro Zeile

    fig3, (ax_hist, ax_prof) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("Radiale Karte — Werteverteilung", fontsize=13)

    ax_hist.hist(valid_vals, bins=64, color="#4488CC", edgecolor="white", linewidth=0.3)
    ax_hist.set_xlabel("r_norm Wert", fontsize=11)
    ax_hist.set_ylabel("Pixel-Anzahl", fontsize=11)
    ax_hist.set_title("Histogramm der radialen Karte (valider Bereich)")
    ax_hist.grid(True, alpha=0.3)

    row_idx_axis = np.arange(N_r)
    ax_prof.plot(row_mean, row_idx_axis, color="#2266AA", linewidth=1.2)
    ax_prof.axhline(N_r - 1, color="red",  linestyle=":", linewidth=1.0,
                    label=f"r_max (Zeile {N_r-1})")
    ax_prof.axhline(0,       color="gray", linestyle=":", linewidth=1.0,
                    label="Zentrum (Zeile 0)")
    ax_prof.set_xlabel("Mittlerer r_norm Wert", fontsize=11)
    ax_prof.set_ylabel("Zeilen-Index", fontsize=11)
    ax_prof.set_title("Zeilenweiser Mittelwert\n(zeigt gültige Zeilen durch Abweichung von 0)")
    ax_prof.invert_yaxis()
    ax_prof.legend(fontsize=9)
    ax_prof.grid(True, alpha=0.3)
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
