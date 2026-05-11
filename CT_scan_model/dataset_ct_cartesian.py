"""PyTorch dataset for cartesian CT diffusion training (image + circular mask).

Returns:
  x:    float tensor [2, CARTESIAN_SIZE, CARTESIAN_SIZE] (image + mask)
  cond: tuple(cat, cont) where
        cat : long tensor  [3] with (cell_format_id, manufacturer_id, chemistry_id)
        cont: float tensor [3] with (slice_depth_relative, voxel_size_um, r_valid_rel)

The mask is 1 inside the estimated cell circle (r_valid) and 0 outside.
Images are resized (no crop) to a fixed square CARTESIAN_SIZE.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _vocab_index(vocab: List[str], value: str) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        return vocab.index("Unknown") if "Unknown" in vocab else 0


def _resize_gray(gray: np.ndarray, size: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for dataset loading.")
    return cv2.resize(gray, (int(size), int(size)), interpolation=cv2.INTER_AREA)


def _scale_geometry(cx: float, cy: float, r_valid: float, w: int, h: int, size: int) -> tuple[float, float, float]:
    # Support non-square originals by scaling x and y separately.
    sx = float(size) / max(1.0, float(w))
    sy = float(size) / max(1.0, float(h))
    cx_s = float(cx) * sx
    cy_s = float(cy) * sy
    # Use the mean scale for radius (robust when images are not square).
    r_s = float(r_valid) * 0.5 * (sx + sy)
    return cx_s, cy_s, r_s


def _circle_mask(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    dist2 = (xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2
    return (dist2 <= float(r) ** 2).astype(np.float32)


@dataclass(frozen=True)
class SampleRef:
    cell_id: str
    image_relpath: str
    slice_depth_relative: float


class BatteryCTCartesianDataset(Dataset):
    def __init__(
        self,
        index_json: str,
        geometry_json: str,
        splits_json: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        with open(index_json, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(geometry_json, "r", encoding="utf-8") as f:
            self.geometry = json.load(f)["cells"]

        self.base_path = self.index["base_path"]
        self.size = int(getattr(project_config, "CARTESIAN_SIZE", 1024))

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

        h, w = gray.shape[:2]
        gray_rs = _resize_gray(gray, size=self.size)

        cx = float(g["cx"])
        cy = float(g["cy"])
        r_valid = float(g["r_valid"])
        cx_s, cy_s, r_s = _scale_geometry(cx, cy, r_valid, w=w, h=h, size=self.size)

        # Clamp radius into a sane range.
        r_s = float(np.clip(r_s, 1.0, float(self.size) / 2.0))

        # Normalize image to [-1, 1]. Input is 0..255.
        img = (gray_rs.astype(np.float32) / 255.0) * 2.0 - 1.0

        mask = _circle_mask(size=self.size, cx=cx_s, cy=cy_s, r=r_s)

        x = torch.from_numpy(np.stack([img, mask], axis=0)).to(dtype=torch.float32)

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

        # r_valid_rel relative to the maximal circle radius in a square image.
        r_valid_rel = float(r_s) / (float(self.size) / 2.0)
        r_valid_rel = float(np.clip(r_valid_rel, 0.0, 1.0))

        cont = torch.tensor(
            [
                float(s.slice_depth_relative),
                float(voxel_size_um),
                float(r_valid_rel),
            ],
            dtype=torch.float32,
        )
        return x, (cat, cont), torch.from_numpy(mask)


class BatteryCTPerCellDataset(Dataset):
    """Cell-level dataset: one sample per cell index.

    Each __getitem__ selects one slice from the given cell_id. This reduces
    redundancy when many slice depths exist per cell.

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
        self.size = int(getattr(project_config, "CARTESIAN_SIZE", 1024))
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
        img_relpath = d["relpath"]
        slice_depth_relative = float(d.get("rel_depth", 0.0))

        img_path = os.path.join(self.base_path, img_relpath)
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for dataset loading.")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")
        h, w = gray.shape[:2]
        gray_rs = _resize_gray(gray, size=self.size)

        cx = float(g["cx"])
        cy = float(g["cy"])
        r_valid = float(g["r_valid"])
        cx_s, cy_s, r_s = _scale_geometry(cx, cy, r_valid, w=w, h=h, size=self.size)
        r_s = float(np.clip(r_s, 1.0, float(self.size) / 2.0))

        img = (gray_rs.astype(np.float32) / 255.0) * 2.0 - 1.0
        mask = _circle_mask(size=self.size, cx=cx_s, cy=cy_s, r=r_s)
        x = torch.from_numpy(np.stack([img, mask], axis=0)).to(dtype=torch.float32)

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
        r_valid_rel = float(np.clip(float(r_s) / (float(self.size) / 2.0), 0.0, 1.0))
        cont = torch.tensor(
            [float(slice_depth_relative), float(voxel_size_um), float(r_valid_rel)],
            dtype=torch.float32,
        )
        return x, (cat, cont), torch.from_numpy(mask)


class BatteryCTSelectedSamplesDataset(Dataset):
    """Dataset over an explicit list of image samples."""

    def __init__(self, index_json: str, geometry_json: str, samples: List[dict]) -> None:
        with open(index_json, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        with open(geometry_json, "r", encoding="utf-8") as f:
            self.geometry = json.load(f)["cells"]

        self.base_path = self.index["base_path"]
        self.size = int(getattr(project_config, "CARTESIAN_SIZE", 1024))

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
        h, w = gray.shape[:2]
        gray_rs = _resize_gray(gray, size=self.size)

        cx = float(g["cx"])
        cy = float(g["cy"])
        r_valid = float(g["r_valid"])
        cx_s, cy_s, r_s = _scale_geometry(cx, cy, r_valid, w=w, h=h, size=self.size)
        r_s = float(np.clip(r_s, 1.0, float(self.size) / 2.0))

        img = (gray_rs.astype(np.float32) / 255.0) * 2.0 - 1.0
        mask = _circle_mask(size=self.size, cx=cx_s, cy=cy_s, r=r_s)
        x = torch.from_numpy(np.stack([img, mask], axis=0)).to(dtype=torch.float32)

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
        r_valid_rel = float(np.clip(float(r_s) / (float(self.size) / 2.0), 0.0, 1.0))
        cont = torch.tensor(
            [float(slice_depth_relative), float(voxel_size_um), float(r_valid_rel)],
            dtype=torch.float32,
        )
        return x, (cat, cont), torch.from_numpy(mask)


class BatteryCTUniformCellsMaxPicturesDataset(Dataset):
    """Training dataset with a fixed global picture budget and uniform cell sampling."""

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
        self.size = int(getattr(project_config, "CARTESIAN_SIZE", 1024))
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
        digest = hashlib.md5(key).digest()  # nosec - not for security
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
        img_relpath = d["relpath"]
        slice_depth_relative = float(d.get("rel_depth", 0.0))

        img_path = os.path.join(self.base_path, img_relpath)
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for dataset loading.")
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Cannot open image: {img_path}")
        h, w = gray.shape[:2]
        gray_rs = _resize_gray(gray, size=self.size)

        cx = float(g["cx"])
        cy = float(g["cy"])
        r_valid = float(g["r_valid"])
        cx_s, cy_s, r_s = _scale_geometry(cx, cy, r_valid, w=w, h=h, size=self.size)
        r_s = float(np.clip(r_s, 1.0, float(self.size) / 2.0))

        img = (gray_rs.astype(np.float32) / 255.0) * 2.0 - 1.0
        mask = _circle_mask(size=self.size, cx=cx_s, cy=cy_s, r=r_s)
        x = torch.from_numpy(np.stack([img, mask], axis=0)).to(dtype=torch.float32)

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
        r_valid_rel = float(np.clip(float(r_s) / (float(self.size) / 2.0), 0.0, 1.0))
        cont = torch.tensor(
            [float(slice_depth_relative), float(voxel_size_um), float(r_valid_rel)],
            dtype=torch.float32,
        )
        return x, (cat, cont), torch.from_numpy(mask)
