"""Erkennung der drei Regionen im polaren CT-Bild: Mandrel / Schichten / Rand.

Ansatz: Pro Winkel (alle 0.5°, also 720 Winkel) wird das radiale
Intensitätsprofil der jeweiligen Polarspalte ausgewertet. Dort wo die
Änderungsrate (1. Ableitung) ein signifikantes Maximum hat, liegt ein
Übergang. Anschließend werden die 720 Punkte auf N_theta Bins interpoliert —
identisch zum Vorgehen in detect_cell_boundary.

Ergebnis: r_mandrel_per_angle[N_theta] und r_ring_per_angle[N_theta],
also winkelabhängige, nicht-kreisförmige Grenzkurven.

Usage
-----
  python test_region_detection.py              # Dateidialog
  python test_region_detection.py image.png    # expliziter Pfad

Figures
-------
Figure 1 — Polarbild mit per-Winkel-Grenzkurven
Figure 2 — Radiales Profil + Gradient für einen einzelnen Winkel (Debug)
Figure 3 — Kartesische Rückprojektion mit eingezeichneten Konturen
Figure 4 — 3-Klassen-Maske (polar + kartesisch)
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
import matplotlib.patches as mpatches

try:
    import cv2
except ImportError:
    raise SystemExit("OpenCV (cv2) ist erforderlich: pip install opencv-python")

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNUNG] scipy nicht gefunden — verwende einfachen gleitenden Mittelwert.")

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
# Glättung
# ---------------------------------------------------------------------------

def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if HAS_SCIPY:
        return gaussian_filter1d(x.astype(np.float64), sigma=sigma)
    k = max(1, int(sigma * 3))
    return np.convolve(x.astype(np.float64), np.ones(k) / k, mode="same")


# ---------------------------------------------------------------------------
# Per-Winkel Regionen-Detektion
# ---------------------------------------------------------------------------

def detect_regions_per_angle(
    polar_img: np.ndarray,
    padding_mask: np.ndarray,
    r_valid_per_angle: np.ndarray,
    r_scale: float,
    n_trace_angles: int = 720,
    smooth_sigma: float = 5.0,
    ring_search_frac: float = 0.12,
    mandrel_search_frac: float = 0.50,
    min_peak_rel_height: float = 0.10,
) -> dict:
    """Erkennt Mandrel- und Ring-Grenze pro Winkel über Spalten-Gradienten.

    Vorgehen (analog detect_cell_boundary):
    1. Für jeden der n_trace_angles Winkel die zugehörige Polarspalte auslesen
    2. Intensitätsprofil der Spalte glätten und Gradient berechnen
    3. Ring-Grenze:    größtes positives Peak im letzten ring_search_frac
                       des validen Spaltenbereichs
    4. Mandrel-Grenze: erstes signifikantes Peak im ersten mandrel_search_frac
    5. 720 Punkte → N_theta Bins interpolieren (periodisch, wie detect_cell_boundary)

    Returns
    -------
    dict mit:
      r_mandrel_per_angle : float32 [N_theta]
      r_ring_per_angle    : float32 [N_theta]
      trace_angles_rad    : float64 [n_trace_angles]  für Debug-Plots
      mandrel_trace       : float32 [n_trace_angles]  Rohwerte vor Interpolation
      ring_trace          : float32 [n_trace_angles]
      debug_col           : int     Beispiel-Spalte für Figure 2
      debug_profile       : float64 Intensitätsprofil der debug_col
      debug_gradient      : float64 Gradient der debug_col
      debug_r_mandrel     : int
      debug_r_ring        : int
    """
    N_r, N_theta = polar_img.shape

    trace_angles_rad = np.linspace(0.0, 2.0 * math.pi, n_trace_angles, endpoint=False)
    # Abbildung Winkel → Spalten-Index
    trace_cols = (trace_angles_rad / (2.0 * math.pi) * N_theta).astype(int) % N_theta

    mandrel_trace = np.zeros(n_trace_angles, dtype=np.float32)
    ring_trace    = np.zeros(n_trace_angles, dtype=np.float32)

    debug_col      = trace_cols[n_trace_angles // 4]   # 90° als Beispiel
    debug_profile  = None
    debug_gradient = None
    debug_r_m      = 0
    debug_r_r      = 0

    for k, j in enumerate(trace_cols):
        # Valide Zeilen für diese Spalte
        r_valid_j   = float(r_valid_per_angle[j])
        r_valid_idx = min(N_r - 1, max(1, int(round(r_valid_j / r_scale))))

        # Spaltenprofil bis zur validen Grenze
        profile = polar_img[:r_valid_idx, j].astype(np.float64)
        n = len(profile)
        if n < 10:
            ring_trace[k]    = float(r_valid_idx - 1)
            mandrel_trace[k] = 0.0
            continue

        profile_smooth = _smooth(profile, sigma=smooth_sigma)
        grad           = np.gradient(profile_smooth)

        # --- Ring-Grenze: stärkstes positives Peak im letzten ring_search_frac ---
        ring_start = max(0, int(n * (1.0 - ring_search_frac)))
        ring_local = grad[ring_start:]
        ring_trace[k] = float(ring_start + int(np.argmax(ring_local)))

        # --- Mandrel-Grenze: erstes signifikantes Peak im ersten mandrel_search_frac ---
        mandrel_end  = max(2, int(n * mandrel_search_frac))
        mand_region  = grad[:mandrel_end]
        threshold    = float(np.max(np.abs(grad))) * min_peak_rel_height

        # suche das erste lokale Maximum über dem Schwellwert
        found = False
        for i in range(1, mandrel_end - 1):
            if (mand_region[i] > mand_region[i - 1] and
                    mand_region[i] > mand_region[i + 1] and
                    mand_region[i] > threshold):
                mandrel_trace[k] = float(i)
                found = True
                break
        if not found:
            mandrel_trace[k] = float(int(np.argmax(mand_region)))

        # Debug-Spalte speichern
        if j == debug_col:
            debug_profile  = profile_smooth
            debug_gradient = grad
            debug_r_m      = int(mandrel_trace[k])
            debug_r_r      = int(ring_trace[k])

    # --- Interpolation auf N_theta Bins (periodisch) ---
    theta_out    = np.linspace(0.0, 2.0 * math.pi, N_theta, endpoint=False)
    angles_wrap  = np.append(trace_angles_rad, trace_angles_rad[0] + 2.0 * math.pi)

    m_wrap = np.append(mandrel_trace, mandrel_trace[0])
    r_wrap = np.append(ring_trace,    ring_trace[0])

    r_mandrel_per_angle = np.interp(theta_out, angles_wrap, m_wrap).astype(np.float32)
    r_ring_per_angle    = np.interp(theta_out, angles_wrap, r_wrap).astype(np.float32)

    return {
        "r_mandrel_per_angle": r_mandrel_per_angle,
        "r_ring_per_angle":    r_ring_per_angle,
        "trace_angles_rad":    trace_angles_rad,
        "mandrel_trace":       mandrel_trace,
        "ring_trace":          ring_trace,
        "debug_col":           int(debug_col),
        "debug_profile":       debug_profile if debug_profile is not None else np.zeros(10),
        "debug_gradient":      debug_gradient if debug_gradient is not None else np.zeros(10),
        "debug_r_mandrel":     debug_r_m,
        "debug_r_ring":        debug_r_r,
    }


def build_three_class_mask(
    padding_mask: np.ndarray,
    r_mandrel_per_angle: np.ndarray,
    r_ring_per_angle: np.ndarray,
) -> np.ndarray:
    """3-Klassen-Maske: 0=Padding, 1=Mandrel, 2=Schichten, 3=Rand."""
    N_r, N_theta = padding_mask.shape
    rows = np.arange(N_r, dtype=np.float32)[:, np.newaxis]   # [N_r, 1]
    cls  = np.zeros((N_r, N_theta), dtype=np.uint8)
    valid = padding_mask > 0.5
    cls[valid & (rows <  r_mandrel_per_angle[np.newaxis, :])                         ] = 1
    cls[valid & (rows >= r_mandrel_per_angle[np.newaxis, :])
              & (rows <  r_ring_per_angle   [np.newaxis, :])                         ] = 2
    cls[valid & (rows >= r_ring_per_angle   [np.newaxis, :])                         ] = 3
    return cls


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

    # --- Bild laden ---
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Bild konnte nicht geladen werden: {img_path}")
    H, W = gray.shape
    print(f"Originalgröße: {W}×{H} px")

    # --- Zellgrenze detektieren ---
    cx, cy, r_valid_per_angle, edge_pts_xy, _ = detect_cell_boundary(
        gray, N_theta=N_theta, kernel_size=25, n_trace_angles=720,
    )
    r_max   = float(np.max(r_valid_per_angle))
    r_scale = r_max / max(1.0, float(N_r - 1))

    # --- Polartransformation ---
    img_norm = (gray.astype(np.float32) / 255.0) * 2.0 - 1.0
    polar_img, padding_mask, r_max_out = cart_to_polar_boundary(
        img_norm, cx, cy, r_valid_per_angle, N_r, N_theta, pad_val,
    )

    # --- Per-Winkel Regionen-Detektion ---
    det = detect_regions_per_angle(
        polar_img, padding_mask, r_valid_per_angle,
        r_scale=r_scale, n_trace_angles=720,
        smooth_sigma=5.0, ring_search_frac=0.12,
        mandrel_search_frac=0.50,
    )
    r_mandrel_per_angle = det["r_mandrel_per_angle"]
    r_ring_per_angle    = det["r_ring_per_angle"]

    print(f"Mandrel-Grenze:  {r_mandrel_per_angle.mean():.1f} ± {r_mandrel_per_angle.std():.1f} Zeilen  "
          f"(r={r_mandrel_per_angle.mean()*r_scale:.1f} ± {r_mandrel_per_angle.std()*r_scale:.1f} px)")
    print(f"Ring-Grenze:     {r_ring_per_angle.mean():.1f} ± {r_ring_per_angle.std():.1f} Zeilen  "
          f"(r={r_ring_per_angle.mean()*r_scale:.1f} ± {r_ring_per_angle.std()*r_scale:.1f} px)")

    # --- 3-Klassen-Maske ---
    cls_mask = build_three_class_mask(padding_mask, r_mandrel_per_angle, r_ring_per_angle)

    # Darstellung
    theta_deg   = np.linspace(0, 360, N_theta, endpoint=False)
    r_valid_rows = r_valid_per_angle / r_scale
    polar_u8    = np.clip((polar_img + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    ext         = [0, 360, N_r, 0]

    color_map = np.array([
        [0,   0,   0,   0  ],   # 0 Padding
        [60,  60, 220, 170 ],   # 1 Mandrel (blau)
        [60, 170,  60, 100 ],   # 2 Schichten (grün)
        [220, 70,  70, 200 ],   # 3 Rand (rot)
    ], dtype=np.uint8)
    cls_rgba = color_map[cls_mask].astype(np.float32) / 255.0

    # -------------------------------------------------------------------------
    # Figure 1 — Polarbild mit per-Winkel-Grenzkurven
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 6))
    fig1.suptitle(
        f"Per-Winkel Regionen-Erkennung  "
        f"(Mandrel ±{r_mandrel_per_angle.std():.1f} Zeilen, "
        f"Ring ±{r_ring_per_angle.std():.1f} Zeilen)",
        fontsize=13,
    )

    axes1[0].imshow(polar_u8, cmap="gray", vmin=0, vmax=255,
                    aspect="auto", extent=ext)
    axes1[0].plot(theta_deg, r_mandrel_per_angle, color="dodgerblue", linewidth=1.8,
                  label=f"Mandrel-Grenze (μ={r_mandrel_per_angle.mean():.0f})")
    axes1[0].plot(theta_deg, r_ring_per_angle,    color="tomato",     linewidth=1.8,
                  label=f"Ring-Grenze (μ={r_ring_per_angle.mean():.0f})")
    axes1[0].plot(theta_deg, r_valid_rows,         color="lime",       linewidth=1.2,
                  linestyle="--", label="Zellgrenze")
    axes1[0].set_title("Polarbild + per-Winkel-Grenzkurven\n(nicht kreisförmig)")
    axes1[0].set_xlabel("Winkel θ [°]")
    axes1[0].set_ylabel("Radius r [Zeilen-Index]")
    axes1[0].legend(fontsize=9)

    axes1[1].imshow(polar_u8, cmap="gray", vmin=0, vmax=255,
                    aspect="auto", extent=ext)
    axes1[1].imshow(cls_rgba, aspect="auto", extent=ext)
    axes1[1].plot(theta_deg, r_mandrel_per_angle, color="dodgerblue", linewidth=1.5)
    axes1[1].plot(theta_deg, r_ring_per_angle,    color="tomato",     linewidth=1.5)
    p_m = mpatches.Patch(color=(0.24, 0.24, 0.86, 0.7), label="Mandrel")
    p_s = mpatches.Patch(color=(0.24, 0.67, 0.24, 0.5), label="Schichten")
    p_r = mpatches.Patch(color=(0.86, 0.27, 0.27, 0.8), label="Rand")
    axes1[1].legend(handles=[p_m, p_s, p_r], fontsize=9)
    axes1[1].set_title("3-Klassen-Overlay")
    axes1[1].set_xlabel("Winkel θ [°]")
    axes1[1].set_ylabel("Radius r [Zeilen-Index]")
    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 2 — Interaktiv: Profil + Gradient für wählbaren Winkel
    # -------------------------------------------------------------------------
    from matplotlib.widgets import Slider, TextBox

    # Hilfsfunktion: Profil + erkannte Grenzen für einen Winkel berechnen
    def _profile_for_angle(angle_deg: float):
        j = int(round(float(angle_deg) / 360.0 * N_theta)) % N_theta
        r_valid_j   = float(r_valid_per_angle[j])
        r_valid_idx = min(N_r - 1, max(2, int(round(r_valid_j / r_scale))))
        profile     = polar_img[:r_valid_idx, j].astype(np.float64)
        n = len(profile)
        if n < 4:
            return np.zeros(4), np.zeros(4), 0, 1, j
        prof_smooth = _smooth(profile, sigma=5.0)
        grad        = np.gradient(prof_smooth)
        # Ring
        ring_start  = max(0, int(n * (1.0 - 0.12)))
        r_ring      = ring_start + int(np.argmax(grad[ring_start:]))
        # Mandrel
        mand_end    = max(2, int(n * 0.50))
        threshold   = float(np.max(np.abs(grad))) * 0.10
        r_mandrel   = int(np.argmax(grad[:mand_end]))
        for i in range(1, mand_end - 1):
            if (grad[i] > grad[i - 1] and grad[i] > grad[i + 1]
                    and grad[i] > threshold):
                r_mandrel = i
                break
        return prof_smooth, grad, r_mandrel, r_ring, j

    def _draw_fig2(angle_deg: float):
        prof, grad, r_m, r_r, j = _profile_for_angle(angle_deg)
        n        = len(prof)
        row_axis = np.arange(n)

        ax_prof.cla()
        ax_grad.cla()

        # Profil
        ax_prof.plot(prof, row_axis, color="steelblue", linewidth=2.0,
                     label="Intensitätsprofil (geglättet)")
        ax_prof.axhline(r_m, color="dodgerblue", linewidth=2.0, linestyle="--",
                        label=f"Mandrel (Zeile {r_m})")
        ax_prof.axhline(r_r, color="tomato",     linewidth=2.0, linestyle="--",
                        label=f"Ring    (Zeile {r_r})")
        ax_prof.axhspan(0,   r_m, color="dodgerblue", alpha=0.08)
        ax_prof.axhspan(r_m, r_r, color="green",      alpha=0.06)
        ax_prof.axhspan(r_r, n,   color="tomato",      alpha=0.10)
        ax_prof.set_xlabel("Intensität [-1, 1]", fontsize=10)
        ax_prof.set_ylabel("Zeilen-Index (Radius r)", fontsize=10)
        ax_prof.set_title("Intensitätsprofil")
        ax_prof.invert_yaxis()
        ax_prof.legend(fontsize=8)
        ax_prof.grid(True, alpha=0.3)

        # Gradient
        ax_grad.plot(grad, row_axis, color="darkorange", linewidth=1.5, label="dI/dr")
        ax_grad.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax_grad.axhline(r_m, color="dodgerblue", linewidth=2.0, linestyle="--",
                        label=f"Mandrel-Peak (Zeile {r_m})")
        ax_grad.axhline(r_r, color="tomato",     linewidth=2.0, linestyle="--",
                        label=f"Ring-Peak    (Zeile {r_r})")
        ax_grad.plot(grad[r_m], r_m, "o", color="dodgerblue", markersize=10, zorder=5)
        ax_grad.plot(grad[r_r], r_r, "o", color="tomato",     markersize=10, zorder=5)
        ax_grad.axhspan(0,   r_m, color="dodgerblue", alpha=0.08)
        ax_grad.axhspan(r_m, r_r, color="green",      alpha=0.06)
        ax_grad.axhspan(r_r, n,   color="tomato",      alpha=0.10)
        ax_grad.set_xlabel("dI/dr (Änderungsrate)", fontsize=10)
        ax_grad.set_ylabel("Zeilen-Index (Radius r)", fontsize=10)
        ax_grad.set_title("Gradient → Peak = Übergang")
        ax_grad.invert_yaxis()
        ax_grad.legend(fontsize=8)
        ax_grad.grid(True, alpha=0.3)

        # Vertikale Linie im Polarbild (Figure 1) aktualisieren
        angle_col = j / N_theta * 360.0
        vline1[0].set_xdata([angle_col, angle_col])
        vline2[0].set_xdata([angle_col, angle_col])

        fig2.suptitle(
            f"Interaktiv: Spalte {j}  |  θ = {angle_deg:.1f}°  "
            f"|  Mandrel Zeile {r_m}  |  Ring Zeile {r_r}",
            fontsize=11,
        )
        fig2.canvas.draw_idle()
        fig1.canvas.draw_idle()

    # Figure 2 aufbauen (Platz für Slider + TextBox unten)
    fig2 = plt.figure(figsize=(15, 8))
    fig2.subplots_adjust(bottom=0.22, top=0.90)
    ax_prof = fig2.add_subplot(1, 2, 1)
    ax_grad = fig2.add_subplot(1, 2, 2)

    # Vertikale Winkel-Linie in Figure 1 (beide Polar-Axes)
    vline1 = axes1[0].plot([90, 90], [0, N_r], color="yellow",
                            linewidth=1.5, linestyle=":", zorder=10)
    vline2 = axes1[1].plot([90, 90], [0, N_r], color="yellow",
                            linewidth=1.5, linestyle=":", zorder=10)

    # Initialer Plot bei 90°
    _draw_fig2(90.0)

    # --- Slider ---
    ax_slider = fig2.add_axes([0.12, 0.10, 0.76, 0.03])
    slider = Slider(ax_slider, "Winkel θ [°]", 0.0, 359.5,
                    valinit=90.0, valstep=0.5, color="steelblue")

    # --- TextBox ---
    ax_tb_label = fig2.add_axes([0.38, 0.04, 0.08, 0.04])
    ax_tb_label.axis("off")
    ax_tb_label.text(0.5, 0.5, "Winkel eingeben:", ha="center", va="center", fontsize=9)
    ax_textbox = fig2.add_axes([0.47, 0.04, 0.10, 0.04])
    textbox = TextBox(ax_textbox, "", initial="90.0")

    # Callbacks
    def _on_slider(val):
        textbox.set_val(f"{val:.1f}")   # TextBox sync
        _draw_fig2(float(val))

    def _on_textbox(text):
        try:
            angle = float(text) % 360.0
            slider.set_val(angle)        # Slider sync (löst _on_slider aus)
        except ValueError:
            pass

    slider.on_changed(_on_slider)
    textbox.on_submit(_on_textbox)

    # -------------------------------------------------------------------------
    # Figure 3 — Kartesisch: Original + Konturen
    # -------------------------------------------------------------------------
    # Grenzkurven in kartesische Koordinaten rückprojizieren
    def polar_boundary_to_cart(r_per_angle, r_scale, cx, cy, N_theta):
        """Wandelt r_per_angle[N_theta] → (x,y) Koordinaten im Originalbild."""
        thetas = np.linspace(0.0, 2.0 * math.pi, N_theta, endpoint=False)
        r_px   = r_per_angle * r_scale
        xs = cx + r_px * np.cos(thetas)
        ys = cy + r_px * np.sin(thetas)
        return xs, ys

    xs_m, ys_m = polar_boundary_to_cart(r_mandrel_per_angle, r_scale, cx, cy, N_theta)
    xs_r, ys_r = polar_boundary_to_cart(r_ring_per_angle,    r_scale, cx, cy, N_theta)

    if len(edge_pts_xy) >= 3:
        bx = np.append(edge_pts_xy[:, 0], edge_pts_xy[0, 0])
        by = np.append(edge_pts_xy[:, 1], edge_pts_xy[0, 1])

    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 7))
    fig3.suptitle("Kartesische Ansicht — per-Winkel Regionsgrenzen", fontsize=13)

    axes3[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes3[0].plot(np.append(xs_m, xs_m[0]), np.append(ys_m, ys_m[0]),
                  color="dodgerblue", linewidth=2.0, label="Mandrel-Grenze")
    axes3[0].plot(np.append(xs_r, xs_r[0]), np.append(ys_r, ys_r[0]),
                  color="tomato",     linewidth=2.0, label="Ring-Grenze")
    if len(edge_pts_xy) >= 3:
        axes3[0].plot(bx, by, color="lime", linewidth=1.5, linestyle="--",
                      label="Zellgrenze")
    axes3[0].plot(cx, cy, "w+", markersize=12, markeredgewidth=2)
    axes3[0].legend(fontsize=9, loc="lower right")
    axes3[0].set_title("Original + per-Winkel Grenzkonturen")
    axes3[0].axis("off")

    # 3-Klassen-Maske kartesisch rückprojizieren
    cls_float = cls_mask.astype(np.float32) / 3.0
    cls_cart  = polar_to_cart(cls_float, cx, cy, r_max_out, N_r, N_theta, W, 0.0)
    cmap_cls  = plt.get_cmap("Set1")
    gray_rgb  = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    cls_rgba_c = cmap_cls(cls_cart)
    valid_c    = (cls_cart > 0.01)[..., np.newaxis]
    overlay_c  = np.where(valid_c,
                           0.55 * gray_rgb + 0.45 * cls_rgba_c[..., :3],
                           gray_rgb)
    axes3[1].imshow(np.clip(overlay_c, 0, 1))
    axes3[1].set_title("3-Klassen-Overlay (kartesisch)")
    axes3[1].axis("off")
    p_m2 = mpatches.Patch(color=cmap_cls(1/3), label="Mandrel")
    p_s2 = mpatches.Patch(color=cmap_cls(2/3), label="Schichten")
    p_r2 = mpatches.Patch(color=cmap_cls(3/3), label="Rand")
    axes3[1].legend(handles=[p_m2, p_s2, p_r2], fontsize=9, loc="lower right")
    fig3.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 4 — Variabilität der Grenzkurven über alle Winkel
    # -------------------------------------------------------------------------
    trace_deg = np.degrees(det["trace_angles_rad"])

    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 5))
    fig4.suptitle("Variabilität der Grenzkurven pro Winkel (720 Stützstellen)", fontsize=13)

    axes4[0].fill_between(trace_deg, det["mandrel_trace"],
                          det["mandrel_trace"].mean(), alpha=0.25, color="dodgerblue")
    axes4[0].plot(trace_deg, det["mandrel_trace"], color="dodgerblue", linewidth=1.0,
                  label=f"Mandrel  μ={det['mandrel_trace'].mean():.1f}  σ={det['mandrel_trace'].std():.1f}")
    axes4[0].axhline(det["mandrel_trace"].mean(), color="dodgerblue",
                     linewidth=1.5, linestyle="--")
    axes4[0].set_xlabel("Winkel θ [°]")
    axes4[0].set_ylabel("Zeilen-Index")
    axes4[0].set_title("Mandrel-Grenze pro Winkel")
    axes4[0].set_xlim(0, 360)
    axes4[0].legend(fontsize=9)
    axes4[0].grid(True, alpha=0.3)

    axes4[1].fill_between(trace_deg, det["ring_trace"],
                          det["ring_trace"].mean(), alpha=0.25, color="tomato")
    axes4[1].plot(trace_deg, det["ring_trace"], color="tomato", linewidth=1.0,
                  label=f"Ring  μ={det['ring_trace'].mean():.1f}  σ={det['ring_trace'].std():.1f}")
    axes4[1].axhline(det["ring_trace"].mean(), color="tomato",
                     linewidth=1.5, linestyle="--")
    axes4[1].set_xlabel("Winkel θ [°]")
    axes4[1].set_ylabel("Zeilen-Index")
    axes4[1].set_title("Ring-Grenze pro Winkel")
    axes4[1].set_xlim(0, 360)
    axes4[1].legend(fontsize=9)
    axes4[1].grid(True, alpha=0.3)
    fig4.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
