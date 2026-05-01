"""Backward-compatible wrapper.

Prefer:
  python -m model.CT_scan_model.scripts.build_cell_index
"""

from .scripts.build_cell_index import *  # noqa: F401,F403
