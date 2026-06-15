"""Visualisierung des V6-Uebergangsmechanismus fuer die Region-Weight-Karte.

Dieses Skript bildet exakt die alte (V6) Logik aus
CT_scan_model/dataset_ct_polar.py::_compute_region_weights nach und stellt die
einzelnen Schritte grafisch dar.  Im Gegensatz zur V7-Variante
(test_regionMap.py, per-Winkel + blockweise konstant) ist V6:

  * RADIAL-SYMMETRISCH  -> Mandrel-/Ring-Grenze sind je EINE Zeile (= Kreis im
    kartesischen Bild), nicht eine pro Winkel wiggelnde Kurve.
  * UEBERGANGS-basiert  -> nur eine schmale Gauss-Bande um den Uebergang wird
    hochgewichtet (nicht die ganze Region).

V6-Pipeline (Schritt fuer Schritt visualisiert):
  1. row_mean       : mittlere Intensitaet pro Zeile (nur gueltige Pixel)
  2. row_mean_smooth: gleitender Mittelwert (k=15)
  3. grad           : Gradient des geglaetteten Profils
  4. Mandrel-Zeile  : erster signifikanter positiver Peak in den ersten 50 %
  5. Ring-Zeile     : r_valid_row (= aeussere Zellgrenze, ~ N_r-1)
  6. weight         : 1 + Gauss-Boost(Mandrel) + Gauss-Boost(Ring)

Usage
-----
  python test_v6_transition.py              # Dateidialog
  python test_v6_transition.py image.png    # expliziter Pfad

Figures
-------
Figure 1 - V6 1D-Detektions-Pipeline (row_mean -> grad -> Gewichtsprofil)
Figure 2 - Polarbild + Gewichtskarte mit Mandrel-/Ring-Zeile
Figure 3 - Kartesische Rueckprojektion (Kreise = radiale Symmetrie von V6)
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
# V6 Region-Weight-Berechnung (gibt zusaetzlich die Zwischenschritte zurueck)
# ---------------------------------------------------------------------------

def compute_region_weights_v6(
    polar_img: np.ndarray,
    padding_mask: np.ndarray,
    r_valid_row: int,
    N_r: int,
    w_mandrel: float = 3.0,
    w_ring: float = 8.0,
    sigma_mandrel: float = 12.0,
    sigma_ring: float = 6.0,
) -> dict:
    """Exakte Nachbildung der V6 _compute_region_weights, plus Zwischenschritte.

    Returns dict mit:
      weights         [N_r, N_theta]  finale Gewichtskarte
      row_mean        [N_r]           rohes Zeilen-Intensitaetsprofil
      row_mean_smooth [N_r]           geglaettetes Profil (k=15)
      grad            [N_r]           Gradient des geglaetteten Profils
      w_profile       [N_r]           1D-Gewichtsprofil (vor Maskierung)
      r_mandrel_row   int             detektierte Mandrel-Zeile
      r_ring_row      int             Ring-Zeile (= r_valid_row)
      mand_end        int             Suchgrenze (50 % von r_valid_row)
      threshold       float           Peak-Schwelle (0.10 * max|grad|)
    """
    # 1. Zeilenweise mittlere Intensitaet (nur gueltige Pixel pro Zeile)
    row_mean = np.zeros(N_r, dtype=np.float64)
    for i in range(N_r):
        valid = polar_img[i, padding_mask[i] > 0.5]
        row_mean[i] = float(np.mean(valid)) if len(valid) > 0 else 0.0

    # 2. Glaettung (gleitender Mittelwert, ohne scipy)
    k = 15
    kernel = np.ones(k, dtype=np.float64) / k
    row_mean_smooth = np.convolve(row_mean, kernel, mode="same")

    # 3. Gradient
    grad = np.gradient(row_mean_smooth)

    # 4. Mandrel-Grenze: erster signifikanter positiver Peak in den ersten 50 %
    mand_end = max(2, int(r_valid_row * 0.5))
    threshold = float(np.max(np.abs(grad))) * 0.10
    r_mandrel_row = int(np.argmax(grad[:mand_end]))
    for i in range(1, mand_end - 1):
        if (grad[i] > grad[i - 1] and grad[i] > grad[i + 1] and grad[i] > threshold):
            r_mandrel_row = i
            break

    # 5./6. Gauss-Boosts um Mandrel- und Ring-Zeile
    rows = np.arange(N_r, dtype=np.float32)
    w = np.ones(N_r, dtype=np.float32)
    w += (w_mandrel - 1.0) * np.exp(-((rows - r_mandrel_row) / sigma_mandrel) ** 2)
    w += (w_ring    - 1.0) * np.exp(-((rows - r_valid_row  ) / sigma_ring   ) ** 2)

    weights = (w[:, np.newaxis] * padding_mask).astype(np.float32)

    return {
        "weights": weights,
        "row_mean": row_mean,
        "row_mean_smooth": row_mean_smooth,
        "grad": grad,
        "w_profile": w,
        "r_mandrel_row": int(r_mandrel_row),
        "r_ring_row": int(r_valid_row),
        "mand_end": int(mand_end),
        "threshold": float(threshold),
    }


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

    # V6-Hyperparameter (Defaults aus _compute_region_weights)
    w_mandrel = 3.0
    w_ring = 8.0
    sigma_mandrel = 12.0
    sigma_ring = 6.0

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

    # V6: r_valid_row = round(r_max / r_scale) -> praktisch N_r - 1
    r_valid_row = min(N_r - 1, max(1, int(round(r_max_out / r_scale))))

    # --- V6 Region-Weight-Karte + Zwischenschritte ---
    res = compute_region_weights_v6(
        polar_img, padding_mask, r_valid_row, N_r,
        w_mandrel=w_mandrel, w_ring=w_ring,
        sigma_mandrel=sigma_mandrel, sigma_ring=sigma_ring,
    )
    region_weights = res["weights"]
    r_mandrel_row = res["r_mandrel_row"]
    r_ring_row    = res["r_ring_row"]
    mand_end      = res["mand_end"]
    threshold     = res["threshold"]

    print(f"r_valid_row (Ring-Zeile): {r_ring_row}")
    print(f"Mandrel-Zeile detektiert: {r_mandrel_row}  (Suchbereich 0..{mand_end})")
    print(f"Peak-Schwelle: {threshold:.5f}")
    valid_weights = region_weights[padding_mask > 0.5]
    print(f"Gewichtskarte: min={valid_weights.min():.3f}  "
          f"max={valid_weights.max():.3f}  mean={valid_weights.mean():.3f}")

    # Rueckprojektion -> kartesisch
    weights_cart = polar_to_cart(region_weights, cx, cy, r_max_out, N_r, N_theta, W, pad_value=0.0)

    polar_u8 = np.clip((polar_img + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    vmax = max(w_mandrel, w_ring)
    ext = [0, 360, N_r, 0]   # imshow extent (origin='upper')
    rows_axis = np.arange(N_r)

    # -------------------------------------------------------------------------
    # Figure 1 - V6 1D-Detektions-Pipeline
    # -------------------------------------------------------------------------
    fig1, (axA, axB, axC) = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle("V6 Uebergangs-Mechanismus - 1D-Pipeline (radial-symmetrisch)", fontsize=13)

    # (A) row_mean roh vs. geglaettet
    axA.plot(rows_axis, res["row_mean"], color="#999999", linewidth=1.0, label="row_mean (roh)")
    axA.plot(rows_axis, res["row_mean_smooth"], color="#1f77b4", linewidth=2.0,
             label="row_mean_smooth (k=15)")
    axA.axvline(r_mandrel_row, color="dodgerblue", linestyle="--", linewidth=1.5,
                label=f"Mandrel-Zeile = {r_mandrel_row}")
    axA.axvline(r_ring_row, color="tomato", linestyle="--", linewidth=1.5,
                label=f"Ring-Zeile = {r_ring_row}")
    axA.axvspan(0, mand_end, color="gold", alpha=0.12, label=f"Mandrel-Suchbereich (0..{mand_end})")
    axA.set_title("Schritt 1-2: Zeilen-Intensitaetsprofil")
    axA.set_xlabel("Zeilen-Index (Radius r)")
    axA.set_ylabel("mittlere Intensitaet [-1, 1]")
    axA.legend(fontsize=8)
    axA.grid(True, alpha=0.3)

    # (B) Gradient + Schwelle + Mandrel-Peak
    axB.plot(rows_axis, res["grad"], color="#2ca02c", linewidth=1.5, label="grad(row_mean_smooth)")
    axB.axhline(threshold, color="purple", linestyle=":", linewidth=1.5,
                label=f"Schwelle = 0.10*max|grad| = {threshold:.4f}")
    axB.axhline(0.0, color="black", linewidth=0.6)
    axB.axvline(r_mandrel_row, color="dodgerblue", linestyle="--", linewidth=1.5,
                label=f"erster Peak > Schwelle = {r_mandrel_row}")
    axB.axvspan(0, mand_end, color="gold", alpha=0.12, label=f"Suchbereich (0..{mand_end})")
    axB.plot([r_mandrel_row], [res["grad"][r_mandrel_row]], "o", color="dodgerblue", markersize=8)
    axB.set_title("Schritt 3-4: Gradient -> Mandrel-Peak-Detektion")
    axB.set_xlabel("Zeilen-Index (Radius r)")
    axB.set_ylabel("Gradient")
    axB.legend(fontsize=8)
    axB.grid(True, alpha=0.3)

    # (C) Resultierendes 1D-Gewichtsprofil (die zwei Gauss-Boosts)
    axC.plot(rows_axis, res["w_profile"], color="#CC4422", linewidth=2.0, label="weight(row)")
    axC.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="Basis = 1.0")
    axC.axvline(r_mandrel_row, color="dodgerblue", linestyle="--", linewidth=1.2,
                label=f"Mandrel-Boost (x{w_mandrel:g}, sigma={sigma_mandrel:g})")
    axC.axvline(r_ring_row, color="tomato", linestyle="--", linewidth=1.2,
                label=f"Ring-Boost (x{w_ring:g}, sigma={sigma_ring:g})")
    axC.set_title("Schritt 5-6: Gewichtsprofil = 1 + Gauss(Mandrel) + Gauss(Ring)")
    axC.set_xlabel("Zeilen-Index (Radius r)")
    axC.set_ylabel("Gewicht")
    axC.legend(fontsize=8)
    axC.grid(True, alpha=0.3)
    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 2 - Polarbild + Gewichtskarte
    # -------------------------------------------------------------------------
    fig2, (axP, axW) = plt.subplots(1, 2, figsize=(17, 7))
    fig2.suptitle("V6 Region-Weight-Karte im Polarraum (Mandrel-/Ring-Zeile = horizontale Linien)",
                  fontsize=13)

    axP.imshow(polar_u8, cmap="gray", vmin=0, vmax=255, aspect="auto", extent=ext)
    axP.axhline(r_mandrel_row, color="dodgerblue", linewidth=1.8, label=f"Mandrel-Zeile ({r_mandrel_row})")
    axP.axhline(r_ring_row, color="tomato", linewidth=1.8, label=f"Ring-Zeile ({r_ring_row})")
    axP.set_title("Polarbild")
    axP.set_xlabel("Winkel theta [Grad]")
    axP.set_ylabel("Radius r [Zeilen-Index]")
    axP.legend(fontsize=9)

    imW = axW.imshow(region_weights, cmap="inferno", vmin=0, vmax=vmax, aspect="auto", extent=ext)
    axW.axhline(r_mandrel_row, color="dodgerblue", linewidth=1.2, linestyle="--")
    axW.axhline(r_ring_row, color="cyan", linewidth=1.2, linestyle="--")
    axW.set_title("Region-Weight-Karte (V6)\n(1 = normal, hell = Uebergang stark gewichtet, 0 = Padding)")
    axW.set_xlabel("Winkel theta [Grad]")
    axW.set_ylabel("Radius r [Zeilen-Index]")
    plt.colorbar(imW, ax=axW, fraction=0.03, pad=0.02, label="weight")
    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 3 - Kartesische Rueckprojektion (V6 = konzentrische Kreise!)
    # -------------------------------------------------------------------------
    fig3, (axO, axWC, axOv) = plt.subplots(1, 3, figsize=(19, 6))
    fig3.suptitle("V6 Rueckprojektion kartesisch - radiale Symmetrie -> KREISE", fontsize=13)

    def _circle(r_row):
        thetas = np.linspace(0.0, 2.0 * math.pi, 361)
        r_px = r_row * r_scale
        return cx + r_px * np.cos(thetas), cy + r_px * np.sin(thetas)

    axO.imshow(gray, cmap="gray", vmin=0, vmax=255)
    if len(edge_pts_xy) >= 3:
        bx = np.append(edge_pts_xy[:, 0], edge_pts_xy[0, 0])
        by = np.append(edge_pts_xy[:, 1], edge_pts_xy[0, 1])
        axO.plot(bx, by, color="lime", linewidth=1.5, linestyle="--", label="Zellgrenze (per-Winkel)")
    xm, ym = _circle(r_mandrel_row)
    xr, yr = _circle(r_ring_row)
    axO.plot(xm, ym, color="dodgerblue", linewidth=1.8, label="Mandrel-Kreis (V6, radial-symm.)")
    axO.plot(xr, yr, color="tomato", linewidth=1.8, label="Ring-Kreis (V6, radial-symm.)")
    axO.legend(fontsize=8, loc="lower right")
    axO.set_title(f"Original ({W}x{H}) + V6-Kreise")
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
