# CT_scan_model (Polar CT DDPM)

This folder contains the **polar CT diffusion model** (UNet + DDPM utilities) and
the training pipeline.

## Quickstart

Activate your virtual environment first:

```powershell
. "C:\Users\larsr\Documents\PythonVenv\Scripts\Activate.ps1"
```

Then run the pipeline steps in order:

```powershell
python -m model.CT_scan_model.scripts.build_cell_index \
  --out model/CT_scan_model/cell_index.json

python -m model.CT_scan_model.scripts.precompute_geometry \
  --index model/CT_scan_model/cell_index.json \
  --out   model/CT_scan_model/cell_geometry.json

python -m model.CT_scan_model.scripts.build_splits \
  --geometry model/CT_scan_model/cell_geometry.json \
  --out      model/CT_scan_model/splits.json

python -m model.CT_scan_model.scripts.train_ct_ddpm \
  --index    model/CT_scan_model/cell_index.json \
  --geometry model/CT_scan_model/cell_geometry.json \
  --splits   model/CT_scan_model/splits.json

## Sampling (generate synthetic images)

Generate synthetic **polar** images (and optional centered **cartesian** images) from a trained checkpoint:

```powershell
python -m model.CT_scan_model.scripts.sample_ct_ddpm \
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

Splits are stored in `model/CT_scan_model/splits.json`.

## Polar representation + padding mask

The model consumes fixed-size polar tensors:

- `POLAR_THETA_BINS = 1024`
- `POLAR_R_MODEL = 704`

Each training input has 2 channels:

1. image (normalized to `[-1, 1]`)
2. mask (1 = valid radius, 0 = padded radius)

The training loss is a **masked MSE** so padded pixels do not dominate gradients.

## Code layout

### Reusable modules

- `modules_polar_ct.py` – UNet for polar CT with conditioning
- `diffusion_polar.py` – mask-aware DDPM forward/reverse process
- `dataset_ct_polar.py` – dataset: png -> polar tensor + mask + conditioning

### CLI entrypoints

Entry points live in `model/CT_scan_model/scripts/`:

- `build_cell_index.py`
- `precompute_geometry.py`
- `build_splits.py`
- `train_ct_ddpm.py`

Wrappers exist in the package root for backwards compatibility.

### Legacy

Legacy scripts are kept under `model/CT_scan_model/legacy/`.

## Notes

- If `torch.cuda.is_available()` is `False` in your venv, training will run on
  CPU. Install a CUDA-enabled PyTorch build to use the GPU.
