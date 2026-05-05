# Polar CT DDPM – Model & Training Pipeline Summary

This document summarizes the current **polar CT diffusion model** implementation in this repository:

- **Model**: `model/CT_scan_model/modules_polar_ct.py` (`UNet_conditional_polar`)
- **Diffusion utilities**: `model/CT_scan_model/diffusion_polar.py`
- **Datasets**: `model/CT_scan_model/dataset_ct_polar.py`
- **Training script**: `model/CT_scan_model/scripts/train_ct_ddpm.py`
- **Sampling script (manual CLI)**: `model/CT_scan_model/scripts/sample_ct_ddpm.py`

> Note: There is also a `legacy/` folder with older scripts that are not part of the current pipeline.

---

## 1) Model setup

### 1.1 Input/Output

The model operates on **polar images** (radius × angle) with a fixed input shape:

- `R = POLAR_R_MODEL` (default: `704`)
- `Theta = POLAR_THETA_BINS` (default: `1024`)

The **UNet input has 2 channels**:

1. **Image channel**: normalized to `[-1, 1]`
2. **Mask channel**: padding/validity mask in `{0, 1}`

The **UNet output has 1 channel**:

- Predicted noise `ε_θ(x_t, t, cond)` with shape `[B, 1, R, Theta]`

### 1.2 Architecture (UNet)

`UNet_conditional_polar` is a UNet with:

- Non-square inputs: `[B, C, R, Theta]`
- **Mixed padding** inside the network:
  - **theta padding** uses circular padding (periodic)
  - **radius padding** uses non-periodic padding (reflect by default)
- Optional gated self-attention (skipped if token budget is exceeded)

Implementation file:

- `model/CT_scan_model/modules_polar_ct.py`

### 1.3 Conditioning (categorical + continuous)

The model supports conditioning as:

```text
cond = (cat, cont)
cat  : LongTensor  [B, 3]
cont : FloatTensor [B, 3]
```

Where:

**Categorical `cat` fields** (IDs, embedded in the model):

1. `cell_format_id` (e.g. 18650/2170/4680)
2. `manufacturer_id` (e.g. Samsung/Vapcell/BYD/HAKADI/EVE)
3. `chemistry_id` (Lithium-ion/Sodium-ion)

**Continuous `cont` fields** (float features):

1. `slice_depth_relative` (0..1)
2. `voxel_size_um` (µm)
3. `r_valid_rel` (0..1), i.e. valid radius fraction

The conditioning is encoded by `ConditionEncoder` (embeddings + MLP) into a vector with the same dimensionality as the time embedding and injected into each Down/Up block.

Configuration:

- Vocabularies and sizes: `model/config.py`
- Continuous dimension: `COND_CONT_DIM` in `model/config.py`

### 1.4 Classifier-Free Guidance (CFG)

CFG requires the model to be trained on both:

- **conditional**: `cond=(cat,cont)`
- **unconditional**: `cond=None`

During training this is enabled via `--p-uncond` (probability to drop conditioning).

During sampling, CFG blends unconditional and conditional predictions:

```text
pred = uncond + cfg_scale * (cond - uncond)
```

Typical interpretation:

- `cfg_scale = 0` → unconditional
- `cfg_scale = 1` → normal conditional
- `cfg_scale > 1` → stronger conditioning

---

## 2) Data preparation and preprocessing

### 2.1 Expected dataset directory structure

The indexing scripts expect a dataset root `BASE_PATH` (set in `model/config.py`) that contains one or more directories named `slices` at arbitrary depth, e.g.:

```text
BASE_PATH/.../<cell_format>/.../slices/<cell_id>/radial_images/*.png
```

Example path:

```text
/data/.../cylindrical/18650/EVE_33V/slices/1767/radial_images/0.0.png
```

The scripts locate all `slices/<cell_id>/radial_images/*.png` folders recursively.

### 2.2 Slice filtering (10% – 90% of cell height)

The indexer filters slices by relative depth:

```text
rel_depth = abs_depth / max_height
```

Where:

- `abs_depth` is parsed from the filename (e.g. `15.0.png` → `15.0`)
- `max_height` is derived from the cell format (18650→65, 2170→70, 4680→80)

Only slices with:

```text
0.1 <= rel_depth <= 0.9
```

are included in training.

Code:

- `model/CT_scan_model/scripts/build_cell_index.py`
- `model/battery_metadata.py`

### 2.3 Per-cell geometry (center + usable radius)

Polar conversion requires a center `(cx, cy)` and usable radius `r_valid`.

This pipeline computes these **per cell_id** once (using a representative slice close to mid-depth) and reuses them for all slices of that cell:

- Otsu thresholding
- morphology close/open
- largest contour
- `minEnclosingCircle` → `(cx, cy, r_valid)`

Code:

- `model/CT_scan_model/scripts/precompute_geometry.py`

### 2.4 Cartesian → Polar conversion

Each PNG is treated as a cartesian cross-section and converted to polar using `cv2.remap`:

- angle bins: `POLAR_THETA_BINS` (fixed to 1024)
- radius bins: `r_use = min(r_valid, POLAR_R_MODEL)`

The produced polar image is then:

- normalized to `[-1, 1]`
- padded to `POLAR_R_MODEL` rows (constant 0)

Code:

- `_to_polar(...)` in `model/CT_scan_model/dataset_ct_polar.py`

---

## 3) Masking (padding mask and non-constant radius)

### 3.1 What the mask means

The mask is a second input channel with:

- `mask[r, θ] = 1` for valid radius rows `r < r_use`
- `mask[r, θ] = 0` for padded rows `r >= r_use`

`r_use` is derived from per-cell `r_valid` (clipped to the model’s radial size).

### 3.2 How masking is used

Masking is used in **two places**:

1. **Model input**: the UNet sees `[image, mask]` and can learn boundary behavior.
2. **Loss masking**: the diffusion loss is computed only in valid pixels:

```text
loss = sum(((pred - noise)^2) * mask) / sum(mask)
```

This prevents the padded area from dominating gradients.

### 3.3 Non-constant radius in real data

In real scans, the detected/usable radius may vary slightly across slice depths.

Current pipeline choice:

- **Per-cell fixed radius**: `r_valid` is estimated once per cell and reused.

Why:

- Stable and fast.
- Avoids per-slice segmentation noise.

If you want to reflect radius variability:

- Option A (more accurate, slower): compute `r_valid` per slice.
- Option B (augmentation): jitter `r_valid_rel` slightly (e.g. ±0.01) during training/sampling.

---

## 4) Training pipeline

### 4.1 Pipeline scripts

Run in order:

1. **Index building**
   - `python -m model.CT_scan_model.scripts.build_cell_index`
   - Output: `model/CT_scan_model/cell_index.json`

2. **Geometry precomputation**
   - `python -m model.CT_scan_model.scripts.precompute_geometry`
   - Output: `model/CT_scan_model/cell_geometry.json`

3. **Splits**
   - `python -m model.CT_scan_model.scripts.build_splits`
   - Output: `model/CT_scan_model/splits.json`

4. **Training**
   - `python -m model.CT_scan_model.scripts.train_ct_ddpm`
   - Output: `runs/ct_scan_model/<timestamp>/...`

### 4.2 Training dataset sampling policy

Training uses a **cell-level dataset**:

- One training sample per cell per epoch
- The slice depth cycles deterministically across epochs

The user selects `--slices-per-cell K`:

- Default epochs derived as `epochs = K`
- Meaning: across the full training run, each cell is seen with up to `K` different slice depths.

### 4.3 Diffusion objective (what is optimized)

For each batch:

1. Sample random diffusion step `t`
2. Create noisy input `x_t` from `x` (only image channel is noised; mask stays fixed)
3. Predict noise `ε_θ(x_t, t, cond)`
4. Compute masked MSE loss

This is the standard DDPM noise prediction objective.

### 4.4 Validation and testing (current setup)

Validation/test are intentionally small and fast:

- For each split (val/test), pick **one cell per format** (18650/2170/4680)
- For each selected cell, choose one **mid-depth** slice (closest to rel_depth 0.5)

This provides a stable, lightweight signal during training.

### 4.5 Outputs and postprocessing

Training outputs are stored under:

```text
runs/ct_scan_model/<timestamp>/
  checkpoint_best.pt
  weights/
    checkpoint_epoch_XXXX.pt
  val_pictures/
    epoch_XXXX_val_*.png
  samples_training/
    epoch_XXXX/
      cond_00_sample_00.png ...
      metadata.json
  run_config.json
  final_metrics.json
```

**checkpoint_best.pt** is written at the root of the run dir.

Periodic checkpoints go into `weights/`.

Validation input pictures are saved into `val_pictures/`.

Qualitative synthetic samples (optional, controlled by `--sample-every`) are saved into `samples_training/`.

---

## 5) How conditions are passed end-to-end

### 5.1 Where conditions come from

- `cell_format`: from path token (18650/2170/4680)
- `manufacturer`: inferred from image size/path rules (`battery_metadata.determine_manufacturer`)
- `chemistry`: derived from manufacturer (`battery_metadata.determine_chemistry`)
- `voxel_size_um`: derived from cell_format (`battery_metadata.determine_voxel_size`)
- `slice_depth_relative`: parsed from filename and normalized by max cell height
- `r_valid_rel`: computed from `r_use / POLAR_R_MODEL`

### 5.2 How they are encoded

- Categorical strings → IDs using vocabs in `model/config.py`
- Continuous values → float tensor `cont`

### 5.3 How the model uses them

The model embeds categorical IDs, passes continuous values through an MLP, fuses them, and injects them into UNet blocks as condition feature maps.

---

## 6) Adding additional conditions (e.g., winding_count, electrode_thickness)

There are two common ways to add new conditions:

### 6.1 Add as continuous features (recommended for numeric measurements)

For `winding_count`, `electrode_thickness`, etc., continuous conditioning is appropriate.

Steps:

1. **Update config**
   - In `model/config.py`, increase `COND_CONT_DIM` to include the new features.
   - Document the order, e.g.:

     ```text
     cont = [slice_depth_relative, voxel_size_um, r_valid_rel, winding_count, electrode_thickness]
     ```

2. **Update model**
   - In `modules_polar_ct.py`, ensure `cond_cont_dim` passed to `ConditionEncoder` matches the new `COND_CONT_DIM`.
   - The `cont_mlp` input layer must accept the new dimension.

3. **Update dataset**
   - In `dataset_ct_polar.py`, compute or load the new values and append to `cont`.
   - Ensure consistent scaling/ranges (normalization is strongly recommended).

4. **Backwards compatibility**
   - Old checkpoints will not load if the condition dimensions change.
   - Consider versioning config / separate runs.

### 6.2 Add as categorical features (only if values are discrete classes)

If you need a new categorical variable:

1. Add vocab list + size in `model/config.py`
2. Add a new embedding in `ConditionEncoder`
3. Extend `cat` tensor shape and update all call sites

This is more invasive than adding continuous features.

### 6.3 Where to compute the new features

You already have analysis scripts in `Data_Ingestion/` such as:

- `winding_count*.py`
- `electrode_thickness*.py`

Recommended integration strategy:

1. Run these scripts offline to produce a per-slice JSON/CSV table.
2. Load the values in the dataset by matching on `(cell_id, filename)`.

This avoids expensive per-iteration computation inside the training loop.

---

## 7) Sampling (manual, after training)

Use:

- `python -m model.CT_scan_model.scripts.sample_ct_ddpm ...`

This script:

- loads `checkpoint_best.pt` (EMA by default)
- samples from 100% noise using the reverse DDPM process
- applies a mask derived from mandatory `--r-valid-rel`
- saves polar PNGs and optional cartesian PNGs (inverse remap, centered)

---

## 8) Practical notes / common pitfalls

- **OpenCV required** (`cv2`) for polar conversion and PNG writing.
- A large number of slices can make slice-based evaluation extremely slow.
- If you change conditioning dimensionality, old checkpoints will not be compatible.
