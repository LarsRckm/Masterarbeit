"""Backward-compatible wrapper.

Prefer:
  python -m model.CT_scan_model.scripts.build_splits
"""

from .scripts.build_splits import *  # noqa: F401,F403
