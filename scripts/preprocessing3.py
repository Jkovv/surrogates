import os
import re
import json
from pathlib import Path

import numpy as np
import pyvista as pv

BASE_LATTICE_DIR = Path("./sim_output/LatticeData")
BASE_OUT_DIR     = Path("./preprocessed_3d")
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINE_NAMES  = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPE_NAMES = ["EC", "NN", "NA", "M1", "M2"]
CELL_TYPE_IDS   = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5}

N_TIMESTEPS  = 101
WINDOW       = 2  # look-back window for LSTM / branch inputs
# NOTE (Issue 18 — scientific rigor report): This window size is not ablated.
# Before publication, run ablation with WINDOW ∈ {1, 2, 3, 5} and report
# validation loss as a function of window size.


def adaptive_clip_percentile(channel: np.ndarray) -> float:
    """
    Adaptive percentile clipping based on excess kurtosis.

    Thresholds:
        kurtosis <  20  →  100.0 %   (raw MinMax)
        kurtosis < 100  →   99.5 %
        kurtosis < 300  →   99.0 %
        kurtosis < 600  →   98.5 %
        kurtosis >= 600 →   98.0 %

    NOTE (Issue 17 — scientific rigor report): These breakpoints are empirical
    heuristics. Before publication, an ablation study comparing different
    clipping strategies (fixed percentiles, IQR-based, log-transform) against
    downstream model metrics is required to justify this choice.
    """
    flat = channel.flatten().astype(np.float64)
    if len(flat) < 4 or flat.std() < 1e-30:
        return 100.0
    mu   = flat.mean()
    sig  = flat.std()
    kurt = float(np.mean(((flat - mu) / sig) ** 4)) - 3.0
    if   kurt <  20: return 100.0
    elif kurt < 100: return  99.5
    elif kurt < 300: return  99.0
    elif kurt < 600: return  98.5
    else:            return  98.0


def scale_channel(channel: np.ndarray, pct: float):
    """
    Clip at `pct` percentile then linearly map to [-1, 1].
    Returns (scaled, c_min, c_max).

    NOTE (Issue 19 — scientific rigor report): Since cytokine concentrations
    are non-negative (c_min=0 always), the [-1, 1] range means all zero values
    map to -1.0. Consider [0, 1] normalization for physics-informed models
    where output activations naturally produce non-negative values.
    """
    c_min = float(channel.min())
    c_max = float(np.percentile(channel, pct))

    if c_max <= c_min:
        c_max = float(channel.max())
    if c_max <= c_min:
        return np.full_like(channel, -1.0, dtype=np.float32), c_min, c_min

    scaled = (np.clip(channel, c_min, c_max) - c_min) / (c_max - c_min) * 2.0 - 1.0
    return scaled.astype(np.float32), c_min, c_max


def process_grid(data_path: Path):
    # loading vtk files (flat directory)
    vtk_files = sorted(
        [f for f in os.listdir(data_path) if f.endswith(".vtk")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )[:N_TIMESTEPS]

    if not vtk_files:
        print(f"  Skipping '{data_path}': no .vtk files found.")
        return

    # Read grid size from first VTK header (DIMENSIONS line)
    first_mesh = pv.read(str(data_path / vtk_files[0]))
    dims = first_mesh.dimensions  # (nx, ny, nz)
    if not (dims[0] == dims[1] == dims[2]):
        print(f"  Skipping '{data_path}': non-cubic DIMENSIONS {dims}")
        return
    G = int(dims[0])
    print(f"  Detected grid {G}x{G}x{G} from {vtk_files[0]} ({len(vtk_files)} timesteps)")

    out_path = BASE_OUT_DIR / f"{G}x{G}x{G}"
    out_path.mkdir(parents=True, exist_ok=True)

    raw_cyt = np.zeros((N_TIMESTEPS, G, G, G, 6), dtype=np.float32)
    masks   = np.zeros((N_TIMESTEPS, G, G, G, 5), dtype=np.float32)

    for i, fname in enumerate(vtk_files):
        mesh = pv.read(str(data_path / fname))
        for j, ck in enumerate(CYTOKINE_NAMES):
            raw_cyt[i, :, :, :, j] = mesh.point_data[ck].reshape(G, G, G, order="F")
        if "CellType" in mesh.point_data:
            ct = mesh.point_data["CellType"].reshape(G, G, G, order="F")
            for j, (_, cid) in enumerate(CELL_TYPE_IDS.items()):
                masks[i, :, :, :, j] = (ct == cid).astype(np.float32)

    # raw physical data - PINN/PI-DeepONet
    # shape: (101, G, G, G, 6)
    np.save(out_path / "Y_raw_phys.npy", raw_cyt)

    # scaling each cytokine independently to [-1, 1]
    scaled    = np.zeros_like(raw_cyt)
    clip_mins = np.zeros(6, dtype=np.float64)
    clip_maxs = np.zeros(6, dtype=np.float64)
    clip_pcts = np.zeros(6, dtype=np.float64)

    for j in range(6):
        pct                   = adaptive_clip_percentile(raw_cyt[..., j])
        s, c_min, c_max       = scale_channel(raw_cyt[..., j], pct)
        scaled[..., j]        = s
        clip_mins[j]          = c_min
        clip_maxs[j]          = c_max
        clip_pcts[j]          = pct

    n = N_TIMESTEPS - WINDOW  # 99 prediction samples

    # shared targets
    # Y_target[i] = scaled field at timestep i+2 (the frame to predict)
    # shape: (99, G, G, G, 6)
    Y_target = scaled[WINDOW:].astype(np.float32)
    np.save(out_path / "Y_target.npy", Y_target)

    # cell-type masks
    Y_masks_spatial = masks[WINDOW:].astype(np.float32)            # (99, G, G, G, 5)
    Y_masks_pinn    = Y_masks_spatial.reshape(n, G * G * G, 5)     # (99, G³, 5)
    np.save(out_path / "Y_masks_spatial.npy", Y_masks_spatial)
    np.save(out_path / "Y_masks_pinn.npy",    Y_masks_pinn)

    # STA-LSTM input and DeepONet / PI-DeepONet branch input
    # cyto_seq[i] = scaled[i], scaled[i+1]   (2 frames of 6 cytokines)
    # mask_seq[i] = masks[i],  masks[i+1]    (2 frames of 5 cell types)
    # combined[i] = concat([cyto_seq[i], mask_seq[i]], axis=-1)  → 11 channels
    # Shape: (99, 2, G, G, G, 11)

    cyto_seq = np.stack(
        [scaled[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32)  # (99, 2, G, G, G, 6)

    mask_seq = np.stack(
        [masks[i : i + WINDOW] for i in range(n)], axis=0
    ).astype(np.float32)  # (99, 2, G, G, G, 5)

    X_combined = np.concatenate([cyto_seq, mask_seq], axis=-1)  # (99, 2, G, G, G, 11)

    np.save(out_path / "X_lstm.npy",   X_combined)  # STA-LSTM
    np.save(out_path / "X_branch.npy", X_combined)  # DeepONet / PI-DeepONet

    # Trunk / collocation coordinates: (x, y, z, t) in [-1, 1]
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, G, dtype=np.float32)
    xx, yy, zz  = np.meshgrid(xs, ys, zs, indexing="ij")
    coords_flat = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # (G³, 3)

    t_norm = np.linspace(-1.0, 1.0, N_TIMESTEPS, dtype=np.float32)[WINDOW:]  # (99,)

    X_trunk = np.zeros((n, G * G * G, 4), dtype=np.float32)
    for i in range(n):
        X_trunk[i, :, :3] = coords_flat
        X_trunk[i, :,  3] = t_norm[i]

    np.save(out_path / "X_trunk.npy",  X_trunk)   # DeepONet / PI-DeepONet trunk
    np.save(out_path / "X_colloc.npy", X_trunk)   # PINN collocation (same array)

    # PINN-specific
    # Y_ic: scaled cytokine field at t=0, used to enforce u(x,y,z,0) = u_0
    # Shape: (G, G, G, 6)
    Y_ic = scaled[0].astype(np.float32)
    np.save(out_path / "Y_ic.npy", Y_ic)

    # Y_bc_mask - Used to enforce Neumann no-flux boundary conditions:
    # ∂u/∂n = 0 on the boundary (matching ABM Neumann BCs).
    # In 3D, the boundary is the 6 faces of the cube.
    bc_mask = np.zeros((G, G, G), dtype=np.float32)
    bc_mask[0, :, :]  = 1.0
    bc_mask[-1, :, :] = 1.0
    bc_mask[:, 0, :]  = 1.0
    bc_mask[:, -1, :] = 1.0
    bc_mask[:, :, 0]  = 1.0
    bc_mask[:, :, -1] = 1.0
    np.save(out_path / "Y_bc_mask.npy", bc_mask)

    # U-Net baseline inputs
    # Collapse the WINDOW frames into channels for a standard 3D conv input:
    #   X_unet[i] = (G, G, G, WINDOW*11) — flattened time window of cytokines + masks
    #   Y_target already serves as the U-Net output target (G, G, G, 6)
    # Shape: (99, G, G, G, 22)
    X_unet = X_combined.transpose(0, 2, 3, 4, 1, 5)              # (99, G, G, G, 2, 11)
    X_unet = X_unet.reshape(n, G, G, G, WINDOW * 11)             # (99, G, G, G, 22)
    np.save(out_path / "X_unet.npy", X_unet.astype(np.float32))
    unet_note = f"X_unet {X_unet.shape}"

    # metadata
    meta = {
        "grid":        G,
        "n_timesteps": N_TIMESTEPS,
        "n_samples":   n,
        "window":      WINDOW,
        "cytokines":   CYTOKINE_NAMES,
        "cell_types":  CELL_TYPE_NAMES,
        "scaling": {
            "method":          "adaptive_percentile_clip_linear_neg1_to_1",
            "feature_range":   [-1, 1],
            "clip_percentile": clip_pcts.tolist(),
            "min":             clip_mins.tolist(),   # all 0.0
            "max":             clip_maxs.tolist(),   # per-cytokine clip upper bound
            "denorm":          "u_phys = (u_scaled + 1) / 2 * max[j]",
        },
        "files": {
            "Y_raw_phys":      "(101,G,G,G,6) raw physical - PINN/PI-DeepONet PDE residuals",
            "Y_target":        "(99,G,G,G,6)  scaled targets - all models",
            "Y_masks_spatial": "(99,G,G,G,5)  cell-type masks - eval + loss weighting",
            "Y_masks_pinn":    "(99,G^3,5)    masks flattened - PINN/PI-DeepONet",
            "X_lstm":          "(99,2,G,G,G,11) cytokine+mask seq - STA-LSTM",
            "X_branch":        "(99,2,G,G,G,11) cytokine+mask seq - DeepONet/PI-DeepONet branch",
            "X_trunk":         "(99,G^3,4)    (x,y,z,t) coords - DeepONet/PI-DeepONet trunk",
            "X_colloc":        "(99,G^3,4)    collocation pts - PINN/PI-DeepONet",
            "Y_ic":            "(G,G,G,6)     initial condition - PINN",
            "Y_bc_mask":       "(G,G,G)       boundary mask - PINN Neumann BCs (6 faces)",
            "X_unet":          f"(99,G,G,G,{WINDOW*11}) multi-channel volumetric input - U-Net baseline",
        },
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(
        f"  {G:>4}x{G}x{G:<4} | "
        f"range [{scaled.min():+.3f}, {scaled.max():+.3f}] | "
        f"clips: { {CYTOKINE_NAMES[j]: f'{clip_pcts[j]:.1f}' for j in range(6)} } | "
        f"{unet_note}"
    )


if __name__ == "__main__":
    print("Preprocessing 3D VTK simulation data...\n")

    # Handle both layouts:
    #   (a) flat:      BASE_LATTICE_DIR/*.vtk
    #   (b) per-grid:  BASE_LATTICE_DIR/<grid_folder>/*.vtk
    entries     = list(BASE_LATTICE_DIR.iterdir())
    has_flat    = any(p.is_file() and p.suffix == ".vtk" for p in entries)
    subdirs     = [p for p in entries if p.is_dir()]

    if has_flat:
        process_grid(BASE_LATTICE_DIR)
    if subdirs:
        for d in sorted(subdirs):
            process_grid(d)
    if not has_flat and not subdirs:
        print(f"  No .vtk files or subfolders found under {BASE_LATTICE_DIR}")

    print("\nDone.")
