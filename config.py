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
BASE_PATH = "/data/Data/projects/GLIMPSE/cylindrical"

# -----------------------------------------------------------------------------
# Polar representation (model input)
# -----------------------------------------------------------------------------

# Fixed angular resolution (theta bins). Theta is treated as periodic.
POLAR_THETA_BINS = 1024

# Fixed radial model size (r bins) used after padding.
# Needs to be divisible by 2**UNET_NUM_DOWNS.
POLAR_R_MODEL = 704

# Optional reference: typical max usable radius in pixels (for checks/plots)
POLAR_R_VALID_REF_MAX = 662

# -----------------------------------------------------------------------------
# Cartesian representation (model input)
# -----------------------------------------------------------------------------

# Fixed square model size used after resizing.
# Needs to be divisible by 2**UNET_NUM_DOWNS.
CARTESIAN_SIZE = 1024

# -----------------------------------------------------------------------------
# UNet / DDPM model parameters (architecture only)
# -----------------------------------------------------------------------------

UNET_NUM_DOWNS = 5

# Input is (image, mask)
UNET_IN_CHANNELS = 2
UNET_OUT_CHANNELS = 1

UNET_BASE_CHANNELS = 16

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

# Continuous condition values expected (slice depth rel, voxel size, r_valid_rel)
COND_CONT_DIM = 3

# Embedding dims for categorical features.
COND_CAT_EMB_DIM = 64

# Condition encoder output dimension. Usually match TIME_EMB_DIM.
COND_EMB_OUT_DIM = TIME_EMB_DIM
