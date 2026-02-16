import os
import numpy as np
import pyvista as pv
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_LATTICE_DIR = Path("./LatticeData")
BASE_OUT_DIR = Path("./preprocessed") 
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPES = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5} 
N_TIMESTEPS = 101

def process_mesh_resolution(folder_name):
    data_path = BASE_LATTICE_DIR / folder_name
    folder_clean = folder_name.replace("LatticeData", "").replace("(", "").replace(")", "").strip()
    out_path = BASE_OUT_DIR / folder_clean
    out_path.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(
        [f for f in os.listdir(data_path) if f.lower().endswith(".vtk")],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )[:N_TIMESTEPS]

    if not vtk_files:
        print(f"Brak plików VTK w {folder_name}")
        return

    sample_mesh = pv.read(str(data_path / vtk_files[0]))
    grid_size = int(np.sqrt(sample_mesh.n_points))
    print(f"Processing: {folder_clean} ({grid_size}x{grid_size})")

    raw_cytokines = np.zeros((N_TIMESTEPS, grid_size, grid_size, len(CYTOKINE_NAMES)), dtype=np.float32)
    cell_masks = np.zeros((N_TIMESTEPS, grid_size, grid_size, len(CELL_TYPES)), dtype=np.float32)
    
    for i, file in enumerate(vtk_files):
        mesh = pv.read(str(data_path / file))
        for j, ck in enumerate(CYTOKINE_NAMES):
            raw_cytokines[i, :, :, j] = mesh.point_data[ck].reshape(grid_size, grid_size, order="F")
        
        if 'CellType' in mesh.point_data:
            ct_data = mesh.point_data['CellType'].reshape(grid_size, grid_size, order="F")
            for j, (cell_name, cell_id) in enumerate(CELL_TYPES.items()):
                cell_masks[i, :, :, j] = (ct_data == cell_id).astype(np.float32)

    scaler = MinMaxScaler(feature_range=(0, 1))
    # flatten to (Samples*H*W, C)
    flat_data = raw_cytokines.reshape(-1, len(CYTOKINE_NAMES))
    scaled_data = scaler.fit_transform(flat_data).reshape(raw_cytokines.shape)

    window = 2
    n_samples = N_TIMESTEPS - window
    
    # LSTM / branch
    X_lstm = np.stack([scaled_data[i-window:i] for i in range(window, N_TIMESTEPS)])
    Y_target = scaled_data[window:]
    
    # trunk (x, y, t)
    raw_coords = np.array(sample_mesh.points).reshape(grid_size, grid_size, 3)[:, :, :2]
    c_min, c_max = raw_coords.min(), raw_coords.max()
    norm_coords = (raw_coords - c_min) / (c_max - c_min)
    t_norm = np.linspace(0, 1, N_TIMESTEPS)
    
    X_trunk = np.zeros((n_samples, grid_size * grid_size, 3), dtype=np.float32)
    for s in range(n_samples):
        X_trunk[s, :, :2] = norm_coords.reshape(-1, 2)
        X_trunk[s, :, 2] = t_norm[s + window]

    # masks 
    Y_masks_spatial = cell_masks[window:]
    Y_masks_pinn = Y_masks_spatial.reshape(n_samples, grid_size * grid_size, len(CELL_TYPES))

    np.save(out_path / "X_lstm.npy", X_lstm)
    np.save(out_path / "X_trunk.npy", X_trunk)
    np.save(out_path / "Y_target.npy", Y_target)
    np.save(out_path / "Y_masks_spatial.npy", Y_masks_spatial)
    np.save(out_path / "Y_masks_pinn.npy", Y_masks_pinn)

    metadata = {
        "res": folder_clean,
        "grid": grid_size,
        "scaling": {"min": scaler.data_min_.tolist(), "max": scaler.data_max_.tolist()},
        "cytokines": CYTOKINE_NAMES,
        "cells": list(CELL_TYPES.keys()),
        "n_samples": n_samples
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Success!")

if __name__ == "__main__":
    if BASE_LATTICE_DIR.exists():
        folders = sorted([d for d in os.listdir(BASE_LATTICE_DIR) if (BASE_LATTICE_DIR / d).is_dir()])
        for f in folders:
            process_mesh_resolution(f)
