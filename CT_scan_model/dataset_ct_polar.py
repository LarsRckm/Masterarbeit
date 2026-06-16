"""PyTorch dataset for polar CT diffusion training.

Mirrors dataset_ct_cartesian.py but applies a polar coordinate transform so
that concentric battery rings become horizontal stripes.

Direct polar transform
----------------------
Images are transformed directly from their original resolution (typically
1300–1400 px square) without first resizing to CARTESIAN_SIZE.  The radial
scale r_max = mean(r_valid_per_angle) is computed per cell so that each cell
fills all N_r rows in the polar image.

Boundary detection
------------------
The cell boundary is detected by Outside→Inside edge-tracing on the Otsu binary
image (see polar_transform.detect_cell_boundary).  No circle is fitted —
the actual, non-circular boundary is used to build the padding mask.

Returns
-------
x    : float tensor [3, POLAR_N_R, POLAR_N_THETA]
         channel 0 — polar image,    normalised to [-1, 1]
         channel 1 — padding mask,   1 inside cell boundary, 0 outside
         channel 2 — radial map,     i/(N_r-1) inside boundary, 0 outside
cond : tuple(cat, cont)
         cat  : long tensor [3] — (cell_format_id, manufacturer_id, chemistry_id)
         cont : float tensor [2] — (slice_depth_relative, r_valid_rel)
                r_valid_rel = r_max / image_half_size
                            = max(r_valid_per_angle) / (original_image_size / 2)
mask : float tensor [POLAR_N_R, POLAR_N_THETA] — 1 inside cell boundary, 0 outside

Padding convention
------------------
Row 0  → centre of battery (r = 0).
Row N_r-1 → r = r_max (= max cell boundary radius in original image pixels).
Pixels outside the detected cell boundary receive POLAR_PAD_VALUE (0.0)
and are masked out so they do not contribute to the loss.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from model import config as project_config
except ImportError:  # pragma: no cover
    import config as project_config  # type: ignore

from .polar_transform import detect_cell_boundary, cart_to_polar_boundary, detect_regions_per_angle
from .dataset_ct_cartesian import _vocab_index


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_and_detect_boundary(
    img_path: str,
    N_theta: int,
) -> tuple[np.ndarray, float, float, np.ndarray, float]:
    """Load a greyscale image at original resolution and detect the cell boundary.

    Returns
    -------
    gray              : uint8 [H, W]  (original resolution)
    cx, cy            : cell centre (pixel coords in original image)
    r_valid_per_angle : float32 [N_theta]
    image_half_size   : float — min(H, W) / 2, used to normalise r_valid_rel
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for dataset loading.")

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")

    image_half_size = float(min(gray.shape[0], gray.shape[1])) / 2.0

    cx, cy, r_valid_per_angle, _, _ = detect_cell_boundary(
        gray, N_theta=N_theta, kernel_size=25,
    )
    return gray, cx, cy, r_valid_per_angle, image_half_size


def _compute_region_weights(
    polar_img: np.ndarray,
    padding_mask: np.ndarray,
    r_valid_per_angle: np.ndarray,
    r_scale: float,
    N_r: int,
    N_theta: int,
    half_width: int = 5,
    w_mandrel: float = 3.0,
    w_ring: float = 8.0,
    w_base: float = 1.0,
) -> np.ndarray:
    """Hybrid per-pixel loss weight map: Mandrel CORRIDOR (edge) + Can BAND (area).

    Both transition curves (Mandrel/Layer and Layer/Can) are detected per angle
    via ``detect_regions_per_angle`` (column-gradient based, mirrors
    ``detect_cell_boundary``'s per-angle approach), since neither transition is
    a perfect circle in general.

    The two regions are treated differently on purpose:

      |rows - r_mandrel_per_angle[theta]| <= half_width  -> w_mandrel  (narrow edge)
      rows >= r_ring_per_angle[theta]   (inside the cell) -> w_ring    (whole can band)
      else (mandrel core, winding layers)                -> w_base
      outside the cell (padding_mask == 0)               -> 0

    Rationale: the Mandrel/Layer transition is a pure edge (the mandrel core
    behind it is homogeneous), so a narrow ±half_width corridor suffices. The
    can/housing, by contrast, must be reproduced bright across its *whole* width
    (not just at the edge), so the entire band from r_ring out to the cell
    boundary is up-weighted. ``half_width`` therefore only applies to the Mandrel
    corridor. On overlap the can band wins (w_ring).

    Returns
    -------
    weights : float32 [N_r, N_theta]
    """
    r_mandrel_per_angle, r_ring_per_angle = detect_regions_per_angle(
        polar_img, padding_mask, r_valid_per_angle, r_scale=r_scale,
    )

    rows = np.arange(N_r, dtype=np.float32)[:, np.newaxis]   # [N_r, 1]

    w = np.full((N_r, N_theta), float(w_base), dtype=np.float32)

    d_mandrel  = np.abs(rows - r_mandrel_per_angle[np.newaxis, :])
    is_housing = rows >= r_ring_per_angle[np.newaxis, :]

    w[d_mandrel <= float(half_width)] = float(w_mandrel)   # Mandrel: narrow corridor
    w[is_housing]                     = float(w_ring)      # Can: whole band

    return (w * padding_mask).astype(np.float32)


def _compute_region_labels(
    polar_img: np.ndarray,
    padding_mask: np.ndarray,
    r_valid_per_angle: np.ndarray,
    r_scale: float,
    N_r: int,
    N_theta: int,
) -> np.ndarray:
    """Per-pixel 3-class region label map (for per-region normalised losses).

    The two per-angle transition curves (r_mandrel, r_ring) partition each
    angular column into three regions:

        0 = padding (outside the cell)
        1 = mandrel   (rows <  r_mandrel_per_angle[theta])
        2 = layers    (r_mandrel <= rows < r_ring)
        3 = housing   (rows >= r_ring, inside the cell)

    Returns
    -------
    labels : float32 [N_r, N_theta]  with values in {0, 1, 2, 3}
    """
    r_mandrel_per_angle, r_ring_per_angle = detect_regions_per_angle(
        polar_img, padding_mask, r_valid_per_angle, r_scale=r_scale,
    )

    rows = np.arange(N_r, dtype=np.float32)[:, np.newaxis]   # [N_r, 1]
    valid = padding_mask > 0.5

    is_mandrel = rows < r_mandrel_per_angle[np.newaxis, :]
    is_housing = rows >= r_ring_per_angle[np.newaxis, :]
    is_layers  = (~is_mandrel) & (~is_housing)

    labels = np.zeros((N_r, N_theta), dtype=np.float32)
    labels[valid & is_mandrel] = 1.0
    labels[valid & is_layers]  = 2.0
    labels[valid & is_housing] = 3.0
    return labels


def _build_sample_tensor(
    gray: np.ndarray,
    cx: float,
    cy: float,
    r_valid_per_angle: np.ndarray,
    N_r: int,
    N_theta: int,
    pad_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Convert an original-resolution greyscale image to a polar tensor.

    Returns
    -------
    x             : float32 [3, N_r, N_theta]
                      ch0 = polar image, ch1 = binary mask, ch2 = radial map
    mask          : float32 [N_r, N_theta]  (1 inside boundary, 0 outside)
    region_labels : float32 [N_r, N_theta]  (0=padding, 1=mandrel, 2=layers, 3=housing)
    r_max         : float — per-cell radial scale in original image pixels
    """
    img_norm = (gray.astype(np.float32) / 255.0) * 2.0 - 1.0

    polar_img, padding_mask, r_max = cart_to_polar_boundary(
        image=img_norm,
        cx=cx,
        cy=cy,
        r_valid_per_angle=r_valid_per_angle,
        N_r=N_r,
        N_theta=N_theta,
        pad_value=pad_value,
    )

    # Radial map: normalised row index inside cell, 0 outside
    row_idx    = np.arange(N_r, dtype=np.float32) / max(1.0, float(N_r - 1))
    radial_map = row_idx[:, np.newaxis] * np.ones((1, N_theta), dtype=np.float32) * padding_mask

    # Region labels (3-class partition) for the per-region normalised losses
    r_scale = r_max / max(1.0, float(N_r - 1))
    region_labels = _compute_region_labels(
        polar_img, padding_mask, r_valid_per_angle, r_scale, N_r, N_theta,
    )

    polar_t   = torch.from_numpy(polar_img).to(dtype=torch.float32)
    mask_t    = torch.from_numpy(padding_mask).to(dtype=torch.float32)
    radial_t  = torch.from_numpy(radial_map).to(dtype=torch.float32)
    labels_t  = torch.from_numpy(region_labels)
    x = torch.stack([polar_t, mask_t, radial_t], dim=0)   # [3, N_r, N_theta]
    return x, mask_t, labels_t, r_max


def _build_conditioning(
    geometry_entry: dict,
    r_max: float,
    image_half_size: float,
    slice_depth_relative: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (cat, cont) conditioning tensors.

    r_valid_rel = r_max / image_half_size — maximum boundary radius as a fraction
    of the image half-size; used as a scalar conditioning feature.
    """
    cell_format = str(geometry_entry.get("cell_format", "Unknown"))
    manufacturer = str(geometry_entry.get("manufacturer", "Unknown"))
    chemistry = str(geometry_entry.get("chemistry", "Unknown"))

    cat = torch.tensor(
        [
            _vocab_index(project_config.CELL_FORMAT_VOCAB, cell_format),
            _vocab_index(project_config.MANUFACTURER_VOCAB, manufacturer),
            _vocab_index(project_config.CHEMISTRY_VOCAB, chemistry),
        ],
        dtype=torch.long,
    )

    r_valid_rel = float(np.clip(r_max / float(image_half_size), 0.0, 1.0))
    cont = torch.tensor(
        [float(slice_depth_relative), float(r_valid_rel)],
        dtype=torch.float32,
    )
    return cat, cont


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleRef:
    cell_id: str
    image_relpath: str
    slice_depth_relative: float


class BatteryCTPolarPerCellDataset(Dataset):
    """Cell-level polar dataset: one slice per cell per epoch.

    Mirrors BatteryCTPerCellDataset from dataset_ct_cartesian.py.
    """

    def __init__(
        self,
        index_json: str,
        geometry_json: str,
        splits_json: str,
        split: str,
        batch_size: int,
        seed: int = 42,
        pad_to_batch: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train/val/test")

        with open(index_json, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(geometry_json, "r", encoding="utf-8") as f:
            self.geometry = json.load(f)["cells"]
        with open(splits_json, "r", encoding="utf-8") as f:
            splits = json.load(f)

        self.base_path = self.index["base_path"]
        self.N_r = int(getattr(project_config, "POLAR_N_R", 512))
        self.N_theta = int(getattr(project_config, "POLAR_N_THETA", 1024))
        self.pad_value = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))
        self.seed = int(seed)
        self._epoch = 1

        self.cell_ids: List[str] = list(splits[f"{split}_cells"])
        if not self.cell_ids:
            raise RuntimeError(f"No cells found for split='{split}'.")

        if pad_to_batch and int(batch_size) > 0:
            rng = random.Random(self.seed + 17)
            while len(self.cell_ids) % int(batch_size) != 0:
                self.cell_ids.append(rng.choice(self.cell_ids))

        allowed = set(self.cell_ids)
        self.cell_to_images: Dict[str, List[dict]] = {}
        for cell_id, cell_info in self.index["cells"].items():
            if cell_id not in allowed:
                continue
            imgs = list(cell_info.get("images", []))
            if not imgs:
                continue
            self.cell_to_images[cell_id] = imgs

        for cid, imgs in self.cell_to_images.items():
            rng = random.Random(self.seed ^ hash(cid))
            rng.shuffle(imgs)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, idx: int):
        cell_id = self.cell_ids[idx]
        g = self.geometry.get(cell_id)
        if g is None:
            raise KeyError(f"Missing geometry for cell_id: {cell_id}. Run precompute_geometry.py")

        imgs = self.cell_to_images.get(cell_id)
        if not imgs:
            raise KeyError(f"No images indexed for cell_id: {cell_id}. Rebuild cell_index.json")

        j = (self._epoch - 1) % len(imgs)
        d = imgs[j]
        img_path = os.path.join(self.base_path, d["relpath"])
        slice_depth_relative = float(d.get("rel_depth", 0.0))

        gray, cx, cy, r_valid_per_angle, image_half_size = _load_and_detect_boundary(
            img_path, self.N_theta,
        )
        x, mask_t, labels_t, r_max = _build_sample_tensor(
            gray, cx, cy, r_valid_per_angle,
            self.N_r, self.N_theta, self.pad_value,
        )
        cat, cont = _build_conditioning(g, r_max, image_half_size, slice_depth_relative)
        return x, (cat, cont), mask_t, labels_t


class BatteryCTPolarSelectedSamplesDataset(Dataset):
    """Polar dataset over an explicit list of image samples.

    Mirrors BatteryCTSelectedSamplesDataset.
    """

    def __init__(self, index_json: str, geometry_json: str, samples: List[dict]) -> None:
        with open(index_json, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(geometry_json, "r", encoding="utf-8") as f:
            self.geometry = json.load(f)["cells"]

        self.base_path = self.index["base_path"]
        self.N_r = int(getattr(project_config, "POLAR_N_R", 512))
        self.N_theta = int(getattr(project_config, "POLAR_N_THETA", 1024))
        self.pad_value = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))

        self.samples = list(samples)
        if not self.samples:
            raise RuntimeError("BatteryCTPolarSelectedSamplesDataset: empty sample list")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        d = self.samples[idx]
        cell_id = d["cell_id"]
        slice_depth_relative = float(d.get("rel_depth", 0.0))

        g = self.geometry.get(cell_id)
        if g is None:
            raise KeyError(f"Missing geometry for cell_id: {cell_id}. Run precompute_geometry.py")

        img_path = os.path.join(self.base_path, d["relpath"])
        gray, cx, cy, r_valid_per_angle, image_half_size = _load_and_detect_boundary(
            img_path, self.N_theta,
        )
        x, mask_t, labels_t, r_max = _build_sample_tensor(
            gray, cx, cy, r_valid_per_angle,
            self.N_r, self.N_theta, self.pad_value,
        )
        cat, cont = _build_conditioning(g, r_max, image_half_size, slice_depth_relative)
        return x, (cat, cont), mask_t, labels_t


class BatteryCTPolarUniformCellsMaxPicturesDataset(Dataset):
    """Polar training dataset with a fixed global picture budget and uniform cell sampling.

    Mirrors BatteryCTUniformCellsMaxPicturesDataset.
    """

    def __init__(
        self,
        index_json: str,
        geometry_json: str,
        splits_json: str,
        split: str,
        max_pictures: int,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train/val/test")
        if int(max_pictures) < 1:
            raise ValueError("max_pictures must be >= 1")

        with open(index_json, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(geometry_json, "r", encoding="utf-8") as f:
            self.geometry = json.load(f)["cells"]
        with open(splits_json, "r", encoding="utf-8") as f:
            splits = json.load(f)

        self.base_path = self.index["base_path"]
        self.N_r = int(getattr(project_config, "POLAR_N_R", 512))
        self.N_theta = int(getattr(project_config, "POLAR_N_THETA", 1024))
        self.pad_value = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))
        self.seed = int(seed)
        self.max_pictures = int(max_pictures)

        self.cell_ids: List[str] = list(splits.get(f"{split}_cells", []))
        if not self.cell_ids:
            raise RuntimeError(f"No cells found for split='{split}'.")

        allowed = set(self.cell_ids)
        self.cell_to_images: Dict[str, List[dict]] = {}
        for cell_id, cell_info in self.index["cells"].items():
            if cell_id not in allowed:
                continue
            imgs = list(cell_info.get("images", []))
            if not imgs:
                continue
            self.cell_to_images[cell_id] = imgs

        n_cells = len(self.cell_ids)
        k = self.max_pictures // n_cells
        r = self.max_pictures % n_cells
        rng = random.Random(self.seed)

        schedule: List[str] = []
        for cid in self.cell_ids:
            schedule.extend([cid] * k)
        if r > 0:
            schedule.extend(rng.sample(self.cell_ids, r))
        rng.shuffle(schedule)
        if len(schedule) != self.max_pictures:
            raise RuntimeError("Internal error: schedule length mismatch")
        self.cell_schedule = schedule

    def __len__(self) -> int:
        return int(self.max_pictures)

    def _stable_seed(self, cell_id: str, idx: int) -> int:
        key = f"{self.seed}|{cell_id}|{int(idx)}".encode("utf-8")
        digest = hashlib.md5(key).digest()  # nosec
        return int.from_bytes(digest[:8], "little", signed=False)

    def __getitem__(self, idx: int):
        cell_id = self.cell_schedule[int(idx)]
        g = self.geometry.get(cell_id)
        if g is None:
            raise KeyError(f"Missing geometry for cell_id: {cell_id}. Run precompute_geometry.py")

        imgs = self.cell_to_images.get(cell_id)
        if not imgs:
            raise KeyError(f"No images indexed for cell_id: {cell_id}. Rebuild cell_index.json")

        rng = random.Random(self._stable_seed(cell_id, int(idx)))
        d = imgs[rng.randrange(len(imgs))]
        img_path = os.path.join(self.base_path, d["relpath"])
        slice_depth_relative = float(d.get("rel_depth", 0.0))

        gray, cx, cy, r_valid_per_angle, image_half_size = _load_and_detect_boundary(
            img_path, self.N_theta,
        )
        x, mask_t, labels_t, r_max = _build_sample_tensor(
            gray, cx, cy, r_valid_per_angle,
            self.N_r, self.N_theta, self.pad_value,
        )
        cat, cont = _build_conditioning(g, r_max, image_half_size, slice_depth_relative)
        return x, (cat, cont), mask_t, labels_t
