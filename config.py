"""Project-wide configuration.

This file is intentionally lightweight and should not import heavy ML
dependencies (torch, etc.).

Model- and dataset-related knobs are centralized here so the training scripts
can stay minimal.
"""

import os

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

# Geben Sie hier den Startpfad an, der die Ordner der einzelnen Zellformate enthält.
# Beispiel: "C:/Data/Battery_CT_Scans"
BASE_PATH = "/hpcwork/cs355501"

# -----------------------------------------------------------------------------
# Cartesian representation (model input)
# -----------------------------------------------------------------------------

# Fixed square model size used after resizing.
# Needs to be divisible by 2**UNET_NUM_DOWNS_CARTESIAN.
CARTESIAN_SIZE = 1024

# -----------------------------------------------------------------------------
# UNet / DDPM model parameters (architecture only)
# -----------------------------------------------------------------------------

# Cartesian UNet downs.
# For CARTESIAN_SIZE=1024:
#   - UNET_NUM_DOWNS_CARTESIAN=2 -> bottleneck 256x256
#   - UNET_NUM_DOWNS_CARTESIAN=3 -> bottleneck 128x128
#   - UNET_NUM_DOWNS_CARTESIAN=5 -> bottleneck 32x32
UNET_NUM_DOWNS_CARTESIAN = 5

# Input is (image, mask)
UNET_IN_CHANNELS = 2
UNET_OUT_CHANNELS = 1

UNET_BASE_CHANNELS = 64
# POLAR_UNET_BASE_CHANNELS = 96  # oder 128

TIME_EMB_DIM = 512

# Self-attention configuration (applied only at small spatial resolutions)
ATTN_ENABLED = True
ATTN_HEADS = 4
ATTN_MAX_TOKENS = 4096  # e.g. 44x64=2816 ok; 88x128=11264 too large

# -----------------------------------------------------------------------------
# Conditioning (categorical + continuous)
# -----------------------------------------------------------------------------

# Canonical vocabularies. Keep an explicit Unknown token for robustness.
CELL_FORMAT_VOCAB = ["18650", "2170", "4680", "Unknown"]
MANUFACTURER_VOCAB = ["Samsung", "Vapcell", "BYD", "HAKADI", "EVE", "Unknown"]
CHEMISTRY_VOCAB = ["Lithium-ion", "Sodium-ion", "Unknown"]

CELL_FORMAT_VOCAB_SIZE = len(CELL_FORMAT_VOCAB)
MANUFACTURER_VOCAB_SIZE = len(MANUFACTURER_VOCAB)
CHEMISTRY_VOCAB_SIZE = len(CHEMISTRY_VOCAB)

# Continuous condition values used by the cartesian pipeline:
#   (slice_depth_relative, r_valid_rel)
COND_CONT_DIM = 2

# Embedding dims for categorical features.
COND_CAT_EMB_DIM = 64

# Condition encoder output dimension. Usually match TIME_EMB_DIM.
COND_EMB_OUT_DIM = TIME_EMB_DIM

# -----------------------------------------------------------------------------
# Polar representation (model input)
# -----------------------------------------------------------------------------

# Polar image dimensions: [N_r (height, radial), N_theta (width, angular)].
# Both must be divisible by 2**UNET_NUM_DOWNS_POLAR.
#   N_r = 512  → 512 / 2^5 = 16  ✓
#   N_theta = 1024 → 1024 / 2^5 = 32  ✓
POLAR_N_R = 512
POLAR_N_THETA = 1024

# Fill value for pixels outside the valid battery ring (r > r_valid).
# Must be outside the normal image range [-1, 1].
POLAR_PAD_VALUE = 0.0

# Polar UNet: input channels = [polar_image, padding_mask].
# Architecture is otherwise identical to the cartesian UNet.
POLAR_UNET_IN_CHANNELS = 3   # polar_image | binary_mask | radial_map
UNET_NUM_DOWNS_POLAR = 5
