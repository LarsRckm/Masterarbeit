"""PyTorch dataset for polar CT diffusion training (image + padding mask).

Returns:
  x:    float tensor [2, POLAR_R_MODEL, POLAR_THETA_BINS] (image + mask)
  cond: tuple(cat, cont) where
        cat : long tensor  [3] with (cell_format_id, manufacturer_id, chemistry_id)
        cont: float tensor [3] with (slice_depth_relative, voxel_size_um, r_valid_rel)

The mask is 1 in valid radius rows, 0 in padded rows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import random

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None

try:
    from model import config as project_config
except ImportError:  # pragma: no cover
    import config as project_config  # type: ignore


def _vocab_index(vocab: List[str], value: str) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        # Fallback to Unknown if present, else 0.
        return vocab.index("Unknown") if "Unknown" in vocab else 0


def _to_polar(gray: np.ndarray, cx: float, cy: float, r_max: int, n_angles: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for polar conversion.")

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.arange(r_max)
    xs = (cx + np.outer(radii, np.cos(angles))).astype(np.float32)
    ys = (cy + np.outer(radii, np.sin(angles))).astype(np.float32)
    polar = cv2.remap(
        gray.astype(np.float32),
        xs,
        ys,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return polar


@dataclass(frozen=True)
class SampleRef:
    cell_id: str
    image_relpath: str
    slice_depth_relative: float


class BatteryCTPolarDataset(Dataset):
    def __init__(
        self,
        index_json: str,
        geometry_json: str,
        splits_json: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        """Create a dataset.

        Args:
            index_json: path to cell_index.json
            geometry_json: path to cell_geometry.json
            splits_json: optional path to splits.json
            split: one of {"train","val","test"} if splits_json is provided
        """
        with open(index_json, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(geometry_json, "r", encoding="utf-8") as f:
            self.geometry = json.load(f)["cells"]

        self.base_path = self.index["base_path"]
        self.theta_bins = int(project_config.POLAR_THETA_BINS)
        self.r_model = int(project_config.POLAR_R_MODEL)

        allowed_cells: Optional[set] = None
        if splits_json is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError("When splits_json is provided, split must be one of: train/val/test")
            with open(splits_json, "r", encoding="utf-8") as f:
                splits = json.load(f)
            allowed_cells = set(splits[f"{split}_cells"])

        self.samples: List[SampleRef] = []
        for cell_id, cell_info in self.index["cells"].items():
            if allowed_cells is not None and cell_id not in allowed_cells:
                continue
            imgs = cell_info.get("images", [])
            for d in imgs:
                self.samples.append(
                    SampleRef(
                        cell_id=cell_id,
                        image_relpath=d["relpath"],
                        slice_depth_relative=float(d.get("rel_depth", 0.0)),
                    )
                )

        if not self.samples:
            raise RuntimeError("No samples found. Check BASE_PATH, index_json, splits, and depth filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        g = self.geometry.get(s.cell_id)
        if g is None:
            raise KeyError(f"Missing geometry for cell_id: {s.cell_id}. Run precompute_geometry.py")

        img_path = os.path.join(self.base_path, s.image_relpath)
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for dataset loading.")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")

        cx = float(g["cx"])
        cy = float(g["cy"])
        r_valid = float(g["r_valid"])

        r_use = int(min(max(1.0, r_valid), float(self.r_model)))
        polar = _to_polar(gray, cx=cx, cy=cy, r_max=r_use, n_angles=self.theta_bins)

        # Normalize to [-1, 1]. Input is 0..255.
        polar = (polar / 255.0) * 2.0 - 1.0

        # Pad/crop radial dimension to r_model.
        img_full = np.zeros((self.r_model, self.theta_bins), dtype=np.float32)
        img_full[:r_use, :] = polar[:r_use, :]

        mask = np.zeros((self.r_model, self.theta_bins), dtype=np.float32)
        mask[:r_use, :] = 1.0

        x = torch.from_numpy(np.stack([img_full, mask], axis=0)).to(dtype=torch.float32)

        # Conditioning
        cell_format = str(g.get("cell_format", "Unknown"))
        manufacturer = str(g.get("manufacturer", "Unknown"))
        chemistry = str(g.get("chemistry", "Unknown"))
        voxel_size_um = g.get("voxel_size_um", 0.0)
        voxel_size_um = 0.0 if voxel_size_um is None else float(voxel_size_um)

        cat = torch.tensor(
            [
                _vocab_index(project_config.CELL_FORMAT_VOCAB, cell_format),
                _vocab_index(project_config.MANUFACTURER_VOCAB, manufacturer),
                _vocab_index(project_config.CHEMISTRY_VOCAB, chemistry),
            ],
            dtype=torch.long,
        )
        cont = torch.tensor(
            [
                float(s.slice_depth_relative),
                float(voxel_size_um),
                float(r_use) / float(self.r_model),
            ],
            dtype=torch.float32,
        )
        cond = (cat, cont)
        return x, cond, torch.from_numpy(mask)


class BatteryCTPerCellDataset(Dataset):
    """Cell-level dataset: one sample per cell index.

    Each __getitem__ selects one slice from the given cell_id. This is intended
    to drastically reduce redundancy when many slice depths exist per cell.

    If you call set_epoch(e), the slice selection becomes deterministic per cell
    (cycling through a per-cell shuffled slice list).

    Note: deterministic cycling is only reliable with num_workers=0.
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
        self.theta_bins = int(project_config.POLAR_THETA_BINS)
        self.r_model = int(project_config.POLAR_R_MODEL)
        self.seed = int(seed)
        self._epoch = 1

        self.cell_ids: List[str] = list(splits[f"{split}_cells"])
        if not self.cell_ids:
            raise RuntimeError(f"No cells found for split='{split}'.")

        if pad_to_batch and int(batch_size) > 0:
            # Pad so each epoch has full batches.
            rng = random.Random(self.seed + 17)
            while len(self.cell_ids) % int(batch_size) != 0:
                self.cell_ids.append(rng.choice(self.cell_ids))

        # Build per-cell slice lists.
        allowed = set(self.cell_ids)
        self.cell_to_images: Dict[str, List[dict]] = {}
        for cell_id, cell_info in self.index["cells"].items():
            if cell_id not in allowed:
                continue
            imgs = list(cell_info.get("images", []))
            if not imgs:
                continue
            self.cell_to_images[cell_id] = imgs

        # Pre-shuffle each cell's slice list deterministically.
        for cid, imgs in self.cell_to_images.items():
            rng = random.Random(self.seed ^ hash(cid))
            rng.shuffle(imgs)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, idx: int):
        import random as _random  # local to avoid global worker-state surprises

        cell_id = self.cell_ids[idx]
        g = self.geometry.get(cell_id)
        if g is None:
            raise KeyError(f"Missing geometry for cell_id: {cell_id}. Run precompute_geometry.py")

        imgs = self.cell_to_images.get(cell_id)
        if not imgs:
            raise KeyError(f"No images indexed for cell_id: {cell_id}. Rebuild cell_index.json")

        # Deterministic per-epoch cycling through a per-cell shuffled list.
        j = (self._epoch - 1) % len(imgs)
        d = imgs[j]

        img_relpath = d["relpath"]
        slice_depth_relative = float(d.get("rel_depth", 0.0))

        img_path = os.path.join(self.base_path, img_relpath)
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for dataset loading.")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")

        cx = float(g["cx"])
        cy = float(g["cy"])
        r_valid = float(g["r_valid"])
        r_use = int(min(max(1.0, r_valid), float(self.r_model)))

        polar = _to_polar(gray, cx=cx, cy=cy, r_max=r_use, n_angles=self.theta_bins)
        polar = (polar / 255.0) * 2.0 - 1.0

        img_full = np.zeros((self.r_model, self.theta_bins), dtype=np.float32)
        img_full[:r_use, :] = polar[:r_use, :]
        mask = np.zeros((self.r_model, self.theta_bins), dtype=np.float32)
        mask[:r_use, :] = 1.0
        x = torch.from_numpy(np.stack([img_full, mask], axis=0)).to(dtype=torch.float32)

        cell_format = str(g.get("cell_format", "Unknown"))
        manufacturer = str(g.get("manufacturer", "Unknown"))
        chemistry = str(g.get("chemistry", "Unknown"))
        voxel_size_um = g.get("voxel_size_um", 0.0)
        voxel_size_um = 0.0 if voxel_size_um is None else float(voxel_size_um)

        cat = torch.tensor(
            [
                _vocab_index(project_config.CELL_FORMAT_VOCAB, cell_format),
                _vocab_index(project_config.MANUFACTURER_VOCAB, manufacturer),
                _vocab_index(project_config.CHEMISTRY_VOCAB, chemistry),
            ],
            dtype=torch.long,
        )
        cont = torch.tensor(
            [
                float(slice_depth_relative),
                float(voxel_size_um),
                float(r_use) / float(self.r_model),
            ],
            dtype=torch.float32,
        )
        return x, (cat, cont), torch.from_numpy(mask)


class BatteryCTSelectedSamplesDataset(Dataset):
    """Dataset over an explicit list of image samples.

    Each sample dict must contain:
      - cell_id (str)
      - relpath (str)  relative to base_path in cell_index.json
      - rel_depth (float)

    This is useful for small deterministic validation/test subsets.
    """

    def __init__(self, index_json: str, geometry_json: str, samples: List[dict]) -> None:
        with open(index_json, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(geometry_json, "r", encoding="utf-8") as f:
            self.geometry = json.load(f)["cells"]

        self.base_path = self.index["base_path"]
        self.theta_bins = int(project_config.POLAR_THETA_BINS)
        self.r_model = int(project_config.POLAR_R_MODEL)

        self.samples = list(samples)
        if not self.samples:
            raise RuntimeError("BatteryCTSelectedSamplesDataset: empty sample list")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        d = self.samples[idx]
        cell_id = d["cell_id"]
        relpath = d["relpath"]
        slice_depth_relative = float(d.get("rel_depth", 0.0))

        g = self.geometry.get(cell_id)
        if g is None:
            raise KeyError(f"Missing geometry for cell_id: {cell_id}. Run precompute_geometry.py")

        img_path = os.path.join(self.base_path, relpath)
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for dataset loading.")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")

        cx = float(g["cx"])
        cy = float(g["cy"])
        r_valid = float(g["r_valid"])
        r_use = int(min(max(1.0, r_valid), float(self.r_model)))

        polar = _to_polar(gray, cx=cx, cy=cy, r_max=r_use, n_angles=self.theta_bins)
        polar = (polar / 255.0) * 2.0 - 1.0

        img_full = np.zeros((self.r_model, self.theta_bins), dtype=np.float32)
        img_full[:r_use, :] = polar[:r_use, :]
        mask = np.zeros((self.r_model, self.theta_bins), dtype=np.float32)
        mask[:r_use, :] = 1.0
        x = torch.from_numpy(np.stack([img_full, mask], axis=0)).to(dtype=torch.float32)

        cell_format = str(g.get("cell_format", "Unknown"))
        manufacturer = str(g.get("manufacturer", "Unknown"))
        chemistry = str(g.get("chemistry", "Unknown"))
        voxel_size_um = g.get("voxel_size_um", 0.0)
        voxel_size_um = 0.0 if voxel_size_um is None else float(voxel_size_um)

        cat = torch.tensor(
            [
                _vocab_index(project_config.CELL_FORMAT_VOCAB, cell_format),
                _vocab_index(project_config.MANUFACTURER_VOCAB, manufacturer),
                _vocab_index(project_config.CHEMISTRY_VOCAB, chemistry),
            ],
            dtype=torch.long,
        )
        cont = torch.tensor(
            [
                float(slice_depth_relative),
                float(voxel_size_um),
                float(r_use) / float(self.r_model),
            ],
            dtype=torch.float32,
        )
        return x, (cat, cont), torch.from_numpy(mask)
