"""Create train/val/test splits by cell folder with stratification fallback.

Primary:
- Split unit is cell_id (no leakage across slices).
- Always stratify by cell_format.
- Additionally stratify by (manufacturer, chemistry) if feasible; fallback automatically.

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; \
  python -m CT_scan_model.scripts.build_splits \
    --geometry CT_scan_model/cell_geometry.json \
    --out      CT_scan_model/splits.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def _assign_by_ratio(items: List[str], ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
    """Deterministic-ish ratio split given a shuffled list.

    Uses mid-point quantiles to allocate items; preserves order of `items`.
    """
    train_r, val_r, test_r = ratios
    if abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1")

    n = len(items)
    train, val, test = [], [], []
    for i, it in enumerate(items):
        u = (i + 0.5) / max(1, n)
        if u < train_r:
            train.append(it)
        elif u < train_r + val_r:
            val.append(it)
        else:
            test.append(it)
    return train, val, test


def _group_key(cell: dict, level: str) -> Tuple:
    if level == "format_man_chem":
        return (cell.get("cell_format", "Unknown"), cell.get("manufacturer", "Unknown"), cell.get("chemistry", "Unknown"))
    if level == "format_man":
        return (cell.get("cell_format", "Unknown"), cell.get("manufacturer", "Unknown"))
    if level == "format":
        return (cell.get("cell_format", "Unknown"),)
    raise ValueError(f"Unknown stratification level: {level}")


def _is_feasible(groups: Dict[Tuple, List[str]]) -> bool:
    # Heuristic: stratification is only meaningful if we have more than one group and
    # no group is extremely tiny (would make splits unstable).
    if len(groups) <= 1:
        return False
    sizes = [len(v) for v in groups.values()]
    # Allow some small groups but avoid all being singletons.
    if max(sizes) <= 2:
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build train/val/test splits (cell-folder level) for CT training.")
    p.add_argument(
        "--geometry",
        default=os.path.join("CT_scan_model", "cell_geometry.json"),
        help="Input cell_geometry.json path",
    )
    p.add_argument(
        "--out",
        default=os.path.join("CT_scan_model", "splits.json"),
        help="Output splits JSON path",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    args = p.parse_args(argv)

    with open(args.geometry, "r", encoding="utf-8") as f:
        geo = json.load(f)
    cells: Dict[str, dict] = geo["cells"]

    rng = random.Random(int(args.seed))
    ratios = (float(args.train), float(args.val), float(args.test))

    # First split by cell_format buckets (required).
    by_format: Dict[str, List[str]] = defaultdict(list)
    for cell_id, cinfo in cells.items():
        by_format[cinfo.get("cell_format", "Unknown")].append(cell_id)

    train_cells: List[str] = []
    val_cells: List[str] = []
    test_cells: List[str] = []
    strat_report: Dict[str, str] = {}

    for cell_format, cell_ids in sorted(by_format.items()):
        rng.shuffle(cell_ids)
        # Attempt stratification within this format bucket.
        chosen_level = "format"  # default

        # Build groups for manufacturer+chemistry (within this format) and test feasibility.
        groups_mc: Dict[Tuple, List[str]] = defaultdict(list)
        for cid in cell_ids:
            c = cells[cid]
            key = (c.get("manufacturer", "Unknown"), c.get("chemistry", "Unknown"))
            groups_mc[key].append(cid)

        if _is_feasible(groups_mc):
            chosen_level = "format_man_chem"
            for g in groups_mc.values():
                rng.shuffle(g)
                tr, va, te = _assign_by_ratio(g, ratios)
                train_cells.extend(tr)
                val_cells.extend(va)
                test_cells.extend(te)
        else:
            # Fallback: manufacturer only.
            groups_m: Dict[str, List[str]] = defaultdict(list)
            for cid in cell_ids:
                groups_m[cells[cid].get("manufacturer", "Unknown")].append(cid)
            if _is_feasible({(k,): v for k, v in groups_m.items()}):
                chosen_level = "format_man"
                for g in groups_m.values():
                    rng.shuffle(g)
                    tr, va, te = _assign_by_ratio(g, ratios)
                    train_cells.extend(tr)
                    val_cells.extend(va)
                    test_cells.extend(te)
            else:
                chosen_level = "format"
                tr, va, te = _assign_by_ratio(cell_ids, ratios)
                train_cells.extend(tr)
                val_cells.extend(va)
                test_cells.extend(te)

        strat_report[cell_format] = chosen_level

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "seed": int(args.seed),
        "ratios": {"train": float(args.train), "val": float(args.val), "test": float(args.test)},
        "stratification": strat_report,
        "train_cells": sorted(train_cells),
        "val_cells": sorted(val_cells),
        "test_cells": sorted(test_cells),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Wrote splits: {args.out}  (train={len(payload['train_cells'])}, val={len(payload['val_cells'])}, test={len(payload['test_cells'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
