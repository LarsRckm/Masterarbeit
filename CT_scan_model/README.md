# CT_scan_model (Cartesian CT DDPM)

This folder contains the **cartesian CT diffusion model** (UNet + DDPM utilities) and
the training pipeline.

## Quickstart

Activate your virtual environment first:

```powershell
. "C:\Users\larsr\Documents\PythonVenv\Scripts\Activate.ps1"
```

Then run the pipeline steps in order:

```powershell
python -m CT_scan_model.scripts.build_cell_index \
  --out CT_scan_model/cell_index.json

python -m CT_scan_model.scripts.precompute_geometry \
  --index CT_scan_model/cell_index.json \
  --out   CT_scan_model/cell_geometry.json

python -m CT_scan_model.scripts.build_splits \
  --geometry CT_scan_model/cell_geometry.json \
  --out      CT_scan_model/splits.json

python -m CT_scan_model.scripts.train_ct_ddpm \
  --index    CT_scan_model/cell_index.json \
  --geometry CT_scan_model/cell_geometry.json \
  --splits   CT_scan_model/splits.json

## Sampling (generate synthetic images)

Generate synthetic **cartesian** images from a trained checkpoint:

```powershell
python -m CT_scan_model.scripts.sample_ct_ddpm \
  --ckpt runs/ct_scan_model/<timestamp>/checkpoint_best.pt \
  --outdir runs/ct_scan_model/<timestamp>/samples \
  --n 8 \
  --cell-format 18650 \
  --manufacturer EVE \
  --chemistry Lithium-ion \
  --slice-depth-relative 0.50 \
  --voxel-size-um 14.4 \
  --r-valid-rel 0.94 \
  --cfg-scale 1.0
```
```

Training outputs are written to:

```text
runs/ct_scan_model/<timestamp>/
```

## Dataset assumptions

The pipeline expects this folder structure under `model/config.py::BASE_PATH`:

```text
BASE_PATH/
  <format_dir>/
    slices/
      <cell_dir>/
        radial_images/
          *.png
```

Only images in `radial_images/` are used.

### Depth filtering (10%–90%)

Slices are filtered by relative depth:

```
rel_depth = abs_depth / max_height
```

where `abs_depth` is extracted from the filename (e.g. `15.0.png`) and
`max_height` comes from the cell format (`18650 -> 65`, `2170 -> 70`, `4680 -> 80`).

Only slices with `0.1 <= rel_depth <= 0.9` are included.

## Train/Val/Test split

- Split unit is the **cell folder** (`cell_id`), so there is no leakage across
  slices from the same cell.
- Split ratios: `0.8 / 0.1 / 0.1`.
- Stratification:
  - always stratify by **cell_format**
  - additionally stratify by `(manufacturer, chemistry)` if feasible; otherwise
    the script falls back automatically.

Splits are stored in `CT_scan_model/splits.json`.

## Cartesian representation + circular mask

The model consumes fixed-size cartesian tensors:

- `CARTESIAN_SIZE = 1024`

Each training input has 2 channels:

1. image (normalized to `[-1, 1]`)
2. mask (1 = inside circle, 0 = outside circle)

The training loss is a **masked MSE** so pixels outside the mask do not dominate gradients.

## Code layout

### Reusable modules

- `modules_cartesian_ct.py` – UNet for cartesian CT with conditioning
- `diffusion_cartesian.py` – mask-aware DDPM forward/reverse process
- `dataset_ct_cartesian.py` – dataset: png -> resized cartesian tensor + circle mask + conditioning

### CLI entrypoints

Entry points live in `CT_scan_model/scripts/`:

- `build_cell_index.py`
- `precompute_geometry.py`
- `build_splits.py`
- `train_ct_ddpm.py`

Wrappers exist in the package root for backwards compatibility.

### Legacy

Legacy scripts are kept under `CT_scan_model/legacy/`.

## Notes

- If `torch.cuda.is_available()` is `False` in your venv, training will run on
  CPU. Install a CUDA-enabled PyTorch build to use the GPU.
