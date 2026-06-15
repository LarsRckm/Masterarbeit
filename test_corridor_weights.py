"""HYBRID Region-Weight-Karte: Mandrel-Korridor (Kante) + Gehaeuse-BAND (Flaeche).

Idee (kombiniert V6-Philosophie + per-Winkel-Detektion + V7-Region fuers Gehaeuse):
  * Detektiere pro Winkel die beiden Uebergaenge (Mandrel<->Schichten,
    Schichten<->Gehaeuse) via detect_regions_per_angle (aus
    test_region_detection.py).
  * Gewichtung:
        Mandrel<->Schichten : nur ein +-HW Korridor (Kante)   -> w_mandrel = 3
        Gehaeuse            : das GANZE Band ab r_ring         -> w_ring    = 8
        sonst (Mandrel-Kern, Schichten)                       -> w_base    = 1
        Padding                                               -> 0

Begruendung: Der Mandrel-Uebergang ist eine reine Kante (Kern dahinter homogen)
-> schmaler Korridor genuegt. Das Gehaeuse muss dagegen ueber die GANZE Schicht
hell reproduziert werden, nicht nur an der Kante -> ganzes Band ab r_ring.
half_width gilt damit nur noch fuer den Mandrel-Korridor.

Im Gegensatz zu:
  * V6 (test_v6_transition.py): radial-symmetrisch, Gauss, beide nur Kante.
  * V7 (dataset_ct_polar, alt): beide ganze Region (auch Mandrel-Kern).

Usage
-----
  python test_corridor_weights.py              # Dateidialog
  python test_corridor_weights.py image.png    # expliziter Pfad

Figures (analog test_v6_transition.py)
--------------------------------------
Figure 1 - 1D-Ansicht fuer eine Debug-Spalte: Profil -> Gradient -> Gewichtsprofil
Figure 2 - Polarbild + Gewichtskarte (Mandrel-Korridor + Gehaeuse-Band)
Figure 3 - Kartesische Rueckprojektion (Original + Korridor/Band, Karte, Overlay)
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
# Korridor-basierte Region-Weight-Karte
# ---------------------------------------------------------------------------

def compute_region_weights_corridor(
    padding_mask: np.ndarray,
    r_mandrel_per_angle: np.ndarray,
    r_ring_per_angle: np.ndarray,
    half_width: int = 5,
    w_mandrel: float = 3.0,
    w_ring: float = 8.0,
    w_base: float = 1.0,
) -> np.ndarray:
    """Hybrid-Gewichtskarte: Mandrel-Korridor (Kante) + Gehaeuse-BAND (Flaeche).

      |rows - r_mandrel_per_angle[theta]| <= half_width  -> w_mandrel (3)  Kante
      rows >= r_ring_per_angle[theta]   (im Gehaeuse)     -> w_ring    (8)  ganzes Band
      sonst (Mandrel-Kern, Wicklungen)                   -> w_base    (1)
      Padding (padding_mask == 0)                        -> 0

    Begruendung: Der Mandrel<->Schichten-Uebergang ist eine reine Kante -> nur
    der schmale Korridor. Das Gehaeuse muss dagegen ueber die GANZE Schicht hell
    reproduziert werden (nicht nur an der Kante) -> das gesamte Band ab r_ring
    wird mit w_ring gewichtet (half_width gilt nur noch fuer den Mandrel).

    Bei Ueberlappung gewinnt das Gehaeuse-Band (w_ring).

    Returns
    -------
    weights : float32 [N_r, N_theta]
    """
    N_r, N_theta = padding_mask.shape
    rows = np.arange(N_r, dtype=np.float32)[:, np.newaxis]   # [N_r, 1]

    w = np.full((N_r, N_theta), float(w_base), dtype=np.float32)

    d_mandrel = np.abs(rows - r_mandrel_per_angle[np.newaxis, :])
    is_housing = rows >= r_ring_per_angle[np.newaxis, :]

    w[d_mandrel <= float(half_width)] = float(w_mandrel)   # Mandrel: schmaler Korridor
    w[is_housing]                     = float(w_ring)      # Gehaeuse: ganzes Band

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

    # Korridor-/Gewichts-Hyperparameter
    HW = 5            # Korridor-Halbbreite in Zeilen
    w_mandrel = 3.0   # Faktor Mandrel<->Schichten-Uebergang
    w_ring = 8.0      # Faktor Schichten<->Gehaeuse-Uebergang
    w_base = 1.0      # Faktor Rest (innerhalb der Zelle)

    # --- Bild laden ---
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
    print(f"Zellzentrum: cx={cx:.1f}, cy={cy:.1f}  r_max={r_max:.1f} px")

    # --- Polartransformation ---
    img_norm = (gray.astype(np.float32) / 255.0) * 2.0 - 1.0
    polar_img, padding_mask, r_max_out = cart_to_polar_boundary(
        img_norm, cx, cy, r_valid_per_angle, N_r, N_theta, pad_val,
    )
    r_scale = r_max_out / max(1.0, float(N_r - 1))

    # --- Per-Winkel Uebergangs-Detektion ---
    det = detect_regions_per_angle(
        polar_img, padding_mask, r_valid_per_angle,
        r_scale=r_scale, n_trace_angles=720,
        smooth_sigma=5.0, ring_search_frac=0.12, mandrel_search_frac=0.50,
    )
    r_mandrel_per_angle = det["r_mandrel_per_angle"]
    r_ring_per_angle    = det["r_ring_per_angle"]

    print(f"Mandrel-Grenze: {r_mandrel_per_angle.mean():.1f} +/- {r_mandrel_per_angle.std():.1f} Zeilen")
    print(f"Ring-Grenze:    {r_ring_per_angle.mean():.1f} +/- {r_ring_per_angle.std():.1f} Zeilen")

    # --- Korridor-Gewichtskarte ---
    region_weights = compute_region_weights_corridor(
        padding_mask, r_mandrel_per_angle, r_ring_per_angle,
        half_width=HW, w_mandrel=w_mandrel, w_ring=w_ring, w_base=w_base,
    )

    n_m = int((region_weights == w_mandrel).sum())
    n_r = int((region_weights == w_ring).sum())
    n_b = int(((region_weights == w_base) & (padding_mask > 0.5)).sum())
    print(f"Gewichtskarte: Mandrel-Korridor(x{w_mandrel:g})={n_m} px, "
          f"Gehaeuse-Band(x{w_ring:g})={n_r} px, Rest(x{w_base:g})={n_b} px")

    # Rueckprojektion -> kartesisch
    weights_cart = polar_to_cart(region_weights, cx, cy, r_max_out, N_r, N_theta, W, pad_value=0.0)

    polar_u8  = np.clip((polar_img + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    theta_deg = np.linspace(0, 360, N_theta, endpoint=False)
    r_valid_rows = r_valid_per_angle / r_scale
    vmax = max(w_mandrel, w_ring)
    ext  = [0, 360, N_r, 0]

    # -------------------------------------------------------------------------
    # Figure 1 - 1D-Ansicht fuer eine Debug-Spalte (Profil -> Gradient -> Gewicht)
    # -------------------------------------------------------------------------
    jcol     = int(det["debug_col"])
    prof     = np.asarray(det["debug_profile"], dtype=np.float64)
    grad     = np.asarray(det["debug_gradient"], dtype=np.float64)
    r_m_col  = float(r_mandrel_per_angle[jcol])
    r_r_col  = float(r_ring_per_angle[jcol])
    n_col    = len(prof)
    rows_col = np.arange(n_col)
    w_col    = region_weights[:, jcol]          # 1D-Gewichtsprofil dieser Spalte
    rows_full = np.arange(N_r)

    fig1, (axA, axB, axC) = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle(f"Korridor-Gewichtung - 1D-Pipeline (Debug-Spalte j={jcol}, "
                  f"theta={jcol/N_theta*360:.1f} Grad)", fontsize=13)

    # (A) Intensitaetsprofil + Mandrel-Korridor + Gehaeuse-Band
    axA.plot(rows_col, prof, color="steelblue", linewidth=2.0, label="Intensitaet (geglaettet)")
    axA.axvspan(max(0, r_m_col - HW), min(n_col, r_m_col + HW), color="dodgerblue", alpha=0.30,
                label=f"Mandrel-Korridor +-{HW}")
    axA.axvspan(max(0, r_r_col), n_col, color="tomato", alpha=0.25,
                label=f"Gehaeuse-Band (x{w_ring:g})")
    axA.axvline(r_m_col, color="dodgerblue", linestyle="--", linewidth=1.5)
    axA.axvline(r_r_col, color="tomato", linestyle="--", linewidth=1.5)
    axA.set_title("Schritt 1: Intensitaetsprofil der Spalte")
    axA.set_xlabel("Zeilen-Index (Radius r)")
    axA.set_ylabel("Intensitaet [-1, 1]")
    axA.legend(fontsize=8)
    axA.grid(True, alpha=0.3)

    # (B) Gradient + detektierte Uebergaenge
    axB.plot(rows_col, grad, color="darkorange", linewidth=1.5, label="dI/dr")
    axB.axhline(0.0, color="black", linewidth=0.6)
    axB.axvspan(max(0, r_m_col - HW), min(n_col, r_m_col + HW), color="dodgerblue", alpha=0.30)
    axB.axvspan(max(0, r_r_col), n_col, color="tomato", alpha=0.25)
    axB.axvline(r_m_col, color="dodgerblue", linestyle="--", linewidth=1.5,
                label=f"Mandrel-Uebergang (Zeile {r_m_col:.0f})")
    axB.axvline(r_r_col, color="tomato", linestyle="--", linewidth=1.5,
                label=f"Gehaeuse-Uebergang (Zeile {r_r_col:.0f})")
    axB.set_title("Schritt 2: Gradient -> Uebergangs-Position")
    axB.set_xlabel("Zeilen-Index (Radius r)")
    axB.set_ylabel("Gradient")
    axB.legend(fontsize=8)
    axB.grid(True, alpha=0.3)

    # (C) Resultierendes 1D-Gewichtsprofil (Boxcar: 1 / 3 / 8)
    axC.plot(rows_full, w_col, color="#CC4422", linewidth=2.0, label="weight(row)")
    axC.axhline(w_base, color="gray", linestyle=":", linewidth=1.0, label=f"Basis = {w_base:g}")
    axC.axvline(r_m_col, color="dodgerblue", linestyle="--", linewidth=1.2,
                label=f"Mandrel-Korridor (x{w_mandrel:g})")
    axC.axvline(r_r_col, color="tomato", linestyle="--", linewidth=1.2,
                label=f"Gehaeuse-Band ab hier (x{w_ring:g})")
    axC.set_title("Schritt 3: Gewichtsprofil = Mandrel-Korridor (3) + Gehaeuse-Band (8)")
    axC.set_xlabel("Zeilen-Index (Radius r)")
    axC.set_ylabel("Gewicht")
    axC.set_ylim(-0.3, vmax + 0.5)
    axC.legend(fontsize=8)
    axC.grid(True, alpha=0.3)
    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 2 - Polarbild + Gewichtskarte
    # -------------------------------------------------------------------------
    fig2, (axP, axW) = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle("Hybrid-Gewichtskarte im Polarraum (Mandrel-Korridor + Gehaeuse-Band)", fontsize=13)

    axP.imshow(polar_u8, cmap="gray", vmin=0, vmax=255, aspect="auto", extent=ext)
    axP.fill_between(theta_deg, r_mandrel_per_angle - HW, r_mandrel_per_angle + HW,
                     color="dodgerblue", alpha=0.30, linewidth=0, label=f"Mandrel-Korridor +-{HW}")
    axP.fill_between(theta_deg, r_ring_per_angle, r_valid_rows,
                     color="tomato", alpha=0.25, linewidth=0, label=f"Gehaeuse-Band (x{w_ring:g})")
    axP.plot(theta_deg, r_mandrel_per_angle, color="dodgerblue", linewidth=1.5)
    axP.plot(theta_deg, r_ring_per_angle, color="tomato", linewidth=1.5, label="Schichten<->Gehaeuse")
    axP.plot(theta_deg, r_valid_rows, color="lime", linewidth=1.0, linestyle="--", label="Zellgrenze")
    axP.set_title("Polarbild + Mandrel-Korridor + Gehaeuse-Band")
    axP.set_xlabel("Winkel theta [Grad]")
    axP.set_ylabel("Radius r [Zeilen-Index]")
    axP.legend(fontsize=8)

    imW = axW.imshow(region_weights, cmap="inferno", vmin=0, vmax=vmax, aspect="auto", extent=ext)
    axW.plot(theta_deg, r_mandrel_per_angle, color="dodgerblue", linewidth=0.8, linestyle="--")
    axW.plot(theta_deg, r_ring_per_angle, color="cyan", linewidth=0.8, linestyle="--")
    axW.set_title(f"Gewichtskarte (Mandrel-Korridor x{w_mandrel:g} / Gehaeuse-Band x{w_ring:g}, Rest x{w_base:g})")
    axW.set_xlabel("Winkel theta [Grad]")
    axW.set_ylabel("Radius r [Zeilen-Index]")
    plt.colorbar(imW, ax=axW, fraction=0.03, pad=0.02, label="weight")
    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 3 - Kartesische Rueckprojektion
    # -------------------------------------------------------------------------
    def _polar_curve_to_cart(r_per_angle):
        thetas = np.linspace(0.0, 2.0 * math.pi, N_theta, endpoint=False)
        r_px = r_per_angle * r_scale
        return cx + r_px * np.cos(thetas), cy + r_px * np.sin(thetas)

    def _annulus_poly(r_in, r_out):
        xi, yi = _polar_curve_to_cart(np.maximum(r_in, 0.0))
        xo, yo = _polar_curve_to_cart(r_out)
        px = np.concatenate([xo, xo[:1], xi[::-1], xi[-1:]])
        py = np.concatenate([yo, yo[:1], yi[::-1], yi[-1:]])
        return px, py

    fig3, (axO, axWC, axOv) = plt.subplots(1, 3, figsize=(19, 6))
    fig3.suptitle("Korridor-Gewichtskarte - Rueckprojektion kartesisch", fontsize=13)

    axO.imshow(gray, cmap="gray", vmin=0, vmax=255)
    pmx, pmy = _annulus_poly(r_mandrel_per_angle - HW, r_mandrel_per_angle + HW)
    prx, pry = _annulus_poly(r_ring_per_angle, r_valid_rows)   # ganzes Gehaeuse-Band
    axO.fill(pmx, pmy, color="dodgerblue", alpha=0.30, linewidth=0, label=f"Mandrel-Korridor (+-{HW})")
    axO.fill(prx, pry, color="tomato", alpha=0.25, linewidth=0, label=f"Gehaeuse-Band (x{w_ring:g})")
    xs_m, ys_m = _polar_curve_to_cart(r_mandrel_per_angle)
    xs_r, ys_r = _polar_curve_to_cart(r_ring_per_angle)
    axO.plot(np.append(xs_m, xs_m[0]), np.append(ys_m, ys_m[0]), color="dodgerblue", linewidth=1.5)
    axO.plot(np.append(xs_r, xs_r[0]), np.append(ys_r, ys_r[0]), color="tomato", linewidth=1.5)
    if len(edge_pts_xy) >= 3:
        bx = np.append(edge_pts_xy[:, 0], edge_pts_xy[0, 0])
        by = np.append(edge_pts_xy[:, 1], edge_pts_xy[0, 1])
        axO.plot(bx, by, color="lime", linewidth=1.2, linestyle="--", label="Zellgrenze")
    axO.legend(fontsize=8, loc="lower right")
    axO.set_title(f"Original ({W}x{H}) + Mandrel-Korridor + Gehaeuse-Band")
    axO.axis("off")

    imWC = axWC.imshow(weights_cart, cmap="inferno", vmin=0, vmax=vmax)
    axWC.set_title("Gewichtskarte (kartesisch)\n0 = Padding -> hell = stark gewichtet")
    axWC.axis("off")
    plt.colorbar(imWC, ax=axWC, fraction=0.03, pad=0.02, label="weight")

    gray_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    cmap_w = plt.get_cmap("inferno")
    w_norm = np.clip(weights_cart / vmax, 0, 1)
    w_rgba = cmap_w(w_norm)
    valid = (weights_cart > 1e-6)[..., np.newaxis]
    overlay = np.where(valid, 0.5 * gray_rgb + 0.5 * w_rgba[..., :3], gray_rgb)
    axOv.imshow(np.clip(overlay, 0, 1))
    axOv.set_title("Overlay: Original + Gewichtskarte")
    axOv.axis("off")
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
