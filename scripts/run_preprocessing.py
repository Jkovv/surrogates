"""
Runner for preprocessing.py that handles:
  - flat LatticeData layout (no grid-named subfolder)
  - custom output directory
Usage:
  python scripts/run_preprocessing.py <scan_iteration_dir>
e.g.:
  python scripts/run_preprocessing.py data/ICCS2026_v2/scan_iteration_0
"""
import sys
import os
import re
import json
from pathlib import Path

# ── inject the scripts/ dir so we can reuse helpers from preprocessing.py ──
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pyvista as pv

from preprocessing import (
    adaptive_clip_percentile,
    scale_channel,
    CYTOKINE_NAMES,
    CELL_TYPE_NAMES,
    CELL_TYPE_IDS,
    N_TIMESTEPS,
    WINDOW,
    GPR_MAX_GRID,
)

def process_iteration(scan_dir: Path):
    lattice_dir = scan_dir / "combi_clean" / "LatticeData"
    out_dir     = scan_dir / "preprocessed"

    if not lattice_dir.exists():
        sys.exit(f"LatticeData not found: {lattice_dir}")

    # Detect grid size from first VTK file
    vtk_files_all = sorted(
        [f for f in os.listdir(lattice_dir) if f.endswith(".vtk")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )[:N_TIMESTEPS]

    if not vtk_files_all:
        sys.exit(f"No VTK files in {lattice_dir}")

    sample = pv.read(str(lattice_dir / vtk_files_all[0]))
    G = sample.dimensions[0]  # 500 for 500x500
    print(f"Detected grid size: {G}x{G}  ({len(vtk_files_all)} timesteps)")

    grid_out = out_dir / f"{G}x{G}"
    grid_out.mkdir(parents=True, exist_ok=True)

    raw_cyt = np.zeros((N_TIMESTEPS, G, G, 6), dtype=np.float32)
    masks   = np.zeros((N_TIMESTEPS, G, G, 5), dtype=np.float32)

    for i, fname in enumerate(vtk_files_all):
        if i % 10 == 0:
            print(f"  Reading timestep {i}/{N_TIMESTEPS} ...", flush=True)
        mesh = pv.read(str(lattice_dir / fname))
        for j, ck in enumerate(CYTOKINE_NAMES):
            raw_cyt[i, :, :, j] = mesh.point_data[ck].reshape(G, G, order="F")
        if "CellType" in mesh.point_data:
            ct = mesh.point_data["CellType"].reshape(G, G, order="F")
            for j, (_, cid) in enumerate(CELL_TYPE_IDS.items()):
                masks[i, :, :, j] = (ct == cid).astype(np.float32)

    np.save(grid_out / "Y_raw_phys.npy", raw_cyt)

    scaled    = np.zeros_like(raw_cyt)
    clip_mins = np.zeros(6, dtype=np.float64)
    clip_maxs = np.zeros(6, dtype=np.float64)
    clip_pcts = np.zeros(6, dtype=np.float64)

    for j in range(6):
        pct                = adaptive_clip_percentile(raw_cyt[:, :, :, j])
        s, c_min, c_max    = scale_channel(raw_cyt[:, :, :, j], pct)
        scaled[:, :, :, j] = s
        clip_mins[j]       = c_min
        clip_maxs[j]       = c_max
        clip_pcts[j]       = pct

    n = N_TIMESTEPS - WINDOW

    Y_target = scaled[WINDOW:].astype(np.float32)
    np.save(grid_out / "Y_target.npy", Y_target)

    Y_masks_spatial = masks[WINDOW:].astype(np.float32)
    Y_masks_pinn    = Y_masks_spatial.reshape(n, G * G, 5)
    np.save(grid_out / "Y_masks_spatial.npy", Y_masks_spatial)
    np.save(grid_out / "Y_masks_pinn.npy",    Y_masks_pinn)

    cyto_seq = np.stack(
        [scaled[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32)
    mask_seq = np.stack(
        [masks[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32)
    X_combined = np.concatenate([cyto_seq, mask_seq], axis=-1)

    np.save(grid_out / "X_lstm.npy",   X_combined)
    np.save(grid_out / "X_branch.npy", X_combined)
    np.save(grid_out / "X_unet.npy",   X_combined)   # UNet uses same array

    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy      = np.meshgrid(xs, ys, indexing="ij")
    coords_flat = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    t_norm      = np.linspace(-1.0, 1.0, N_TIMESTEPS, dtype=np.float32)[WINDOW:]

    X_trunk = np.zeros((n, G * G, 3), dtype=np.float32)
    for i in range(n):
        X_trunk[i, :, :2] = coords_flat
        X_trunk[i, :,  2] = t_norm[i]

    np.save(grid_out / "X_trunk.npy",  X_trunk)
    np.save(grid_out / "X_colloc.npy", X_trunk)

    Y_ic = scaled[0].astype(np.float32)
    np.save(grid_out / "Y_ic.npy", Y_ic)

    bc_mask = np.zeros((G, G), dtype=np.float32)
    bc_mask[0, :]  = 1.0
    bc_mask[-1, :] = 1.0
    bc_mask[:, 0]  = 1.0
    bc_mask[:, -1] = 1.0
    np.save(grid_out / "Y_bc_mask.npy", bc_mask)

    if G <= GPR_MAX_GRID:
        cyto_flat = Y_target.reshape(n, -1)
        mask_flat = Y_masks_spatial.reshape(n, -1)
        X_gpr     = np.concatenate([cyto_flat, mask_flat], axis=-1).astype(np.float32)
        np.save(grid_out / "X_gpr.npy", X_gpr)
        gpr_note = f"X_gpr {X_gpr.shape}"
    else:
        gpr_note = f"X_gpr skipped (G={G} > {GPR_MAX_GRID})"

    meta = {
        "grid":        G,
        "clip_max":    clip_maxs.tolist(),   # top-level shortcut for model scripts
        "n_timesteps": N_TIMESTEPS,
        "n_samples":   n,
        "window":      WINDOW,
        "cytokines":   CYTOKINE_NAMES,
        "cell_types":  CELL_TYPE_NAMES,
        "scaling": {
            "method":          "adaptive_percentile_clip_linear_neg1_to_1",
            "feature_range":   [-1, 1],
            "clip_percentile": clip_pcts.tolist(),
            "min":             clip_mins.tolist(),
            "max":             clip_maxs.tolist(),
            "denorm":          "u_phys = (u_scaled + 1) / 2 * max[j]",
        },
        "files": {
            "Y_raw_phys":      "(101,G,G,6) raw physical",
            "Y_target":        "(99,G,G,6)  scaled targets",
            "Y_masks_spatial": "(99,G,G,5)  cell-type masks",
            "Y_masks_pinn":    "(99,G*G,5)  masks flattened",
            "X_lstm":          "(99,2,G,G,11) cytokine+mask seq",
            "X_branch":        "(99,2,G,G,11) cytokine+mask seq",
            "X_trunk":         "(99,G*G,3)  (x,y,t) coords",
            "X_colloc":        "(99,G*G,3)  collocation pts",
            "Y_ic":            "(G,G,6)     initial condition",
            "Y_bc_mask":       "(G,G)       boundary mask",
        },
    }
    with open(grid_out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(
        f"\n  {G}x{G} done | "
        f"range [{scaled.min():+.3f}, {scaled.max():+.3f}] | "
        f"clips: { {CYTOKINE_NAMES[j]: f'{clip_pcts[j]:.1f}' for j in range(6)} } | "
        f"{gpr_note}"
    )
    print(f"  Output: {grid_out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/run_preprocessing.py <scan_iteration_dir>")
    process_iteration(Path(sys.argv[1]))
