"""Backward-compatible wrapper.

Prefer:
  python -m model.CT_scan_model.scripts.train_ct_ddpm
"""

from .scripts.train_ct_ddpm import *  # noqa: F401,F403
