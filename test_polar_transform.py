"""Visual test for the boundary-aware cart_to_polar transform.

Usage
-----
  python test_polar_transform.py              # file dialog
  python test_polar_transform.py image.png    # explicit path

Figures
-------
Figure 1 — Cartesian view (original resolution)
  Left   : greyscale + detected boundary (actual edge, not a circle) + centre
  Middle : region overlay  green = inside boundary (cell),  red = outside (padding)
  Right  : detected boundary drawn as a polygon overlay

Figure 2 — Boundary shape analysis
  Left   : per-angle radius r(θ) vs. angle — shows how non-circular the boundary is
  Right  : edge points scattered on the image, colour-coded by angle

Figure 3 — Polar view (512×1024)
  Left   : polar image (pad_value clipped to black)
  Middle : padding mask  (white = valid, black = padding)
           The boundary appears as a near-horizontal line with small variance.
  Right  : polar image + red padding overlay + green boundary curve

Figure 4 — Roundtrip check (cart → polar → cart)
  Left   : original image
  Right  : back-projected from polar
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV (cv2) is required. Install via: pip install opencv-python")

import config as project_config
from CT_scan_model.polar_transform import detect_cell_boundary, cart_to_polar_boundary, polar_to_cart


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
            filetypes=[
                ("Image files", "*.png *.tif *.tiff *.bmp *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not path:
            raise SystemExit("Kein Bild ausgewählt.")
        return path
    except ImportError:
        raise SystemExit("tkinter nicht verfügbar. Bildpfad als Argument übergeben.")


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
    N_r        = int(getattr(project_config, "POLAR_N_R", 512))
    N_theta    = int(getattr(project_config, "POLAR_N_THETA", 1024))
    pad_val    = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))

    # --- Load original image (no resize) ---
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Bild konnte nicht geladen werden: {img_path}")

    H, W = gray.shape
    image_half_size = float(min(H, W)) / 2.0
    print(f"Originalgröße: {W}×{H} px  (image_half_size={image_half_size:.0f} px)")

    # --- Detect actual boundary (no circle fitting) ---
    cx, cy, r_valid_per_angle, edge_pts_xy, edge_angles_rad = detect_cell_boundary(
        gray, N_theta=N_theta, kernel_size=25, n_trace_angles=720,
    )

    r_mean  = float(np.mean(r_valid_per_angle))
    r_min   = float(np.min(r_valid_per_angle))
    r_max_b = float(np.max(r_valid_per_angle))
    r_std   = float(np.std(r_valid_per_angle))
    r_valid_rel = r_max_b / image_half_size

    print(f"Zellzentrum      →  cx={cx:.1f}, cy={cy:.1f}  (px in {W}×{H})")
    print(f"Boundary r_mean  →  {r_mean:.1f} px  (std={r_std:.1f}, min={r_min:.1f}, max={r_max_b:.1f})")
    print(f"r_valid_rel      →  {r_valid_rel:.4f}  (r_mean / image_half_size)")
    print(f"Polar dims       →  {N_r} rows × {N_theta} cols")
    print(f"r_scale          →  {r_max_b / (N_r - 1):.3f} px/row  (jede Zeile = r_max/(N_r-1) px)")

    # --- Polar transform (direct from original, r_max computed internally) ---
    img_norm = (gray.astype(np.float32) / 255.0) * 2.0 - 1.0
    polar_img, padding_mask, r_max = cart_to_polar_boundary(
        img_norm, cx, cy, r_valid_per_angle, N_r, N_theta, pad_val,
    )
    print(f"r_max (internal) →  {r_max:.1f} px  (= r_mean, Radialskala)")

    # --- Back-projection ---
    cart_back = polar_to_cart(polar_img, cx, cy, r_max, N_r, N_theta, W, pad_val)

    # --- Build cartesian boundary mask from detected points ---
    if len(edge_pts_xy) >= 3:
        pts_int = edge_pts_xy.astype(np.int32).reshape(-1, 1, 2)
        boundary_mask_cart = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(boundary_mask_cart, [pts_int], 1)
    else:
        boundary_mask_cart = np.zeros((H, W), dtype=np.uint8)

    inside_mask  = boundary_mask_cart > 0
    outside_mask = boundary_mask_cart == 0

    # -------------------------------------------------------------------------
    # Figure 1 — Cartesian view
    # -------------------------------------------------------------------------
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[inside_mask]  = [0.0, 0.8, 0.0, 0.22]
    overlay[outside_mask] = [0.9, 0.0, 0.0, 0.28]

    boundary_xy = edge_pts_xy

    fig1, axes1 = plt.subplots(1, 3, figsize=(19, 6))
    fig1.suptitle("Kartesische Ansicht — tatsächliche Zellgrenze (kein Kreis)", fontsize=13)

    axes1[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes1[0].fill(boundary_xy[:, 0], boundary_xy[:, 1],
                  color="lime", fill=False, linewidth=1.8, label="Detektierte Grenze")
    axes1[0].plot(cx, cy, "r+", markersize=12, markeredgewidth=2.5, label="Zentrum")
    axes1[0].set_title(f"Original ({W}×{H})\nBoundary-Trace (kein Kreis-Fit)")
    axes1[0].legend(fontsize=8, loc="lower right")
    axes1[0].axis("off")

    axes1[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes1[1].imshow(overlay)
    axes1[1].set_title("Regions-Overlay\nGrün = Zellinneres  |  Rot = Padding")
    axes1[1].axis("off")
    green_p = mpatches.Patch(color=(0.0, 0.8, 0.0, 0.6), label="Innerhalb Grenze (Zelle)")
    red_p   = mpatches.Patch(color=(0.9, 0.0, 0.0, 0.6), label="Außerhalb Grenze (Padding)")
    axes1[1].legend(handles=[green_p, red_p], loc="lower right", fontsize=8)

    axes1[2].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes1[2].imshow(overlay)
    closed_bx = np.append(boundary_xy[:, 0], boundary_xy[0, 0])
    closed_by = np.append(boundary_xy[:, 1], boundary_xy[0, 1])
    axes1[2].plot(closed_bx, closed_by, color="white", linewidth=2.0, linestyle="--")
    axes1[2].plot(cx, cy, "r+", markersize=12, markeredgewidth=2.5)
    axes1[2].set_title(f"Boundary-Polygon\ncx={cx:.0f}, cy={cy:.0f}\n"
                       f"r_mean={r_mean:.0f} px  r_std={r_std:.1f} px")
    axes1[2].axis("off")
    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 2 — Boundary shape analysis
    # -------------------------------------------------------------------------
    dx_e = edge_pts_xy[:, 0] - cx
    dy_e = edge_pts_xy[:, 1] - cy
    edge_r_from_centre = np.sqrt(dx_e ** 2 + dy_e ** 2)
    angles_deg = np.degrees(edge_angles_rad)

    fig2, (ax_r, ax_scatter) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Boundary-Analyse: r(θ) pro Winkel", fontsize=13)

    ax_r.fill_between(angles_deg, edge_r_from_centre, alpha=0.20, color="#4488CC")
    ax_r.plot(angles_deg, edge_r_from_centre, linewidth=0.9, color="#2266AA",
              label="Detektierter Radius r(θ)")
    ax_r.axhline(r_max_b, color="#00CC66", linestyle="--", linewidth=1.5,
                 label=f"r_max = {r_max_b:.1f} px")
    ax_r.axhline(image_half_size, color="#FF4444", linestyle=":", linewidth=1.5,
                 label=f"image_half_size = {image_half_size:.0f} px")
    ax_r.set_xlabel("Winkel θ [°]", fontsize=11)
    ax_r.set_ylabel("Radius r [px]", fontsize=11)
    ax_r.set_title("Radius pro Winkel (zeigt Abweichung vom Kreis)")
    ax_r.set_xlim(0, 360)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, alpha=0.3)

    stats_txt = (
        f"n_angles        = {len(edge_pts_xy)}\n"
        f"r_mean          = {r_mean:.1f} px\n"
        f"r_std           = {r_std:.1f} px\n"
        f"r_min           = {r_min:.1f} px\n"
        f"r_max (= scale) = {r_max_b:.1f} px\n"
        f"image_half_size = {image_half_size:.0f} px\n"
        f"r_valid_rel     = {r_valid_rel:.4f}"
    )
    ax_r.text(0.02, 0.98, stats_txt, transform=ax_r.transAxes, fontsize=9,
              verticalalignment="top",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    sc = ax_scatter.scatter(
        edge_pts_xy[:, 0], edge_pts_xy[:, 1],
        c=np.degrees(edge_angles_rad), cmap="hsv",
        s=4, vmin=0, vmax=360,
    )
    ax_scatter.imshow(gray, cmap="gray", vmin=0, vmax=255, alpha=0.5)
    ax_scatter.plot(cx, cy, "w+", markersize=14, markeredgewidth=2.5)
    ax_scatter.set_title(f"Randpunkte ({len(edge_pts_xy)} Punkte), farbkodiert nach Winkel")
    ax_scatter.set_xlim(0, W)
    ax_scatter.set_ylim(H, 0)
    ax_scatter.axis("off")
    cbar = fig2.colorbar(sc, ax=ax_scatter, fraction=0.03, pad=0.02)
    cbar.set_label("Winkel [°]", fontsize=10)
    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 3 — Polar view
    # -------------------------------------------------------------------------
    polar_display = np.clip(polar_img, -1.0, 1.0)
    polar_uint8   = ((polar_display + 1.0) * 0.5 * 255.0).astype(np.uint8)

    polar_rgba = np.zeros((N_r, N_theta, 4), dtype=np.float32)
    polar_rgba[padding_mask < 0.5] = [0.9, 0.0, 0.0, 0.45]

    # Boundary in polar space: r_valid_per_angle[j] / r_scale = row index.
    r_scale = r_max / max(1.0, float(N_r - 1))
    r_valid_rows = r_valid_per_angle / r_scale   # [N_theta]
    theta_axis_deg = np.linspace(0, 360, N_theta, endpoint=False)

    fig3, axes3 = plt.subplots(1, 3, figsize=(19, 5))
    fig3.suptitle("Polarbild  [Zeilen = Radius r,  Spalten = Winkel θ]\n"
                  f"(r_max = max(r_valid_per_angle) = {r_max:.0f} px → Zeile {N_r-1})", fontsize=12)

    ext = [0, 360, N_r, 0]

    axes3[0].imshow(polar_uint8, cmap="gray", vmin=0, vmax=255, aspect="auto", extent=ext)
    axes3[0].set_title("Polarbild\n(pad_value → schwarz)")
    axes3[0].set_xlabel("Winkel θ [°]")
    axes3[0].set_ylabel("Radius r [Zeilen-Index]")

    axes3[1].imshow(padding_mask, cmap="gray", vmin=0, vmax=1, aspect="auto", extent=ext)
    axes3[1].plot(theta_axis_deg, r_valid_rows, color="lime", linewidth=1.5,
                  label="Detektierte Grenze")
    axes3[1].axhline(N_r - 1, color="cyan", linewidth=1.0, linestyle="--",
                     label=f"Zeile {N_r-1} (= r_max)")
    axes3[1].set_title("Padding-Maske\n(weiß = gültig,  schwarz = Padding)")
    axes3[1].set_xlabel("Winkel θ [°]")
    axes3[1].set_ylabel("Radius r [Zeilen-Index]")
    axes3[1].legend(fontsize=8, loc="lower right")

    axes3[2].imshow(polar_uint8, cmap="gray", vmin=0, vmax=255, aspect="auto", extent=ext)
    axes3[2].imshow(polar_rgba, aspect="auto", extent=ext)
    axes3[2].plot(theta_axis_deg, r_valid_rows, color="lime", linewidth=2.0,
                  label="Zellgrenze (tatsächlich)")
    axes3[2].set_title("Polar + Padding-Overlay\nRot = Padding  |  Grün = Zellgrenze")
    axes3[2].set_xlabel("Winkel θ [°]")
    axes3[2].set_ylabel("Radius r [Zeilen-Index]")
    axes3[2].legend(fontsize=8)
    fig3.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 4 — Roundtrip
    # -------------------------------------------------------------------------
    cart_back_u8 = ((np.clip(cart_back, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 6))
    fig4.suptitle("Roundtrip-Check  (cart → polar → cart)", fontsize=13)

    axes4[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes4[0].fill(boundary_xy[:, 0], boundary_xy[:, 1],
                  color="lime", fill=False, linewidth=1.5)
    axes4[0].set_title(f"Original ({W}×{H})")
    axes4[0].axis("off")

    axes4[1].imshow(cart_back_u8, cmap="gray", vmin=0, vmax=255)
    axes4[1].fill(boundary_xy[:, 0], boundary_xy[:, 1],
                  color="lime", fill=False, linewidth=1.5)
    axes4[1].set_title("Rückprojektion aus Polarbild\n(sollte innerhalb der Grenze identisch sein)")
    axes4[1].axis("off")
    fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
