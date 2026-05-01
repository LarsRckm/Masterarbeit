"""Backward-compatible wrapper.

Prefer:
  python -m model.CT_scan_model.scripts.precompute_geometry
"""

from .scripts.precompute_geometry import *  # noqa: F401,F403
