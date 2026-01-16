import os
import numpy as np
import pyvista as pv
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

BASE_LATTICE_DIR = Path("./LatticeData")
BASE_OUT_DIR = Path("./preprocessed")

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPES = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5} 

N_CYTOKINES = len(CYTOKINE_NAMES)
N_CELLS = len(CELL_TYPES)
N_TIMESTEPS = 101

def process_mesh_resolution(folder_name):
    data_path = BASE_LATTICE_DIR / folder_name
    
    folder_clean = folder_name.replace("LatticeData", "").replace("(", "").replace(")", "").strip()
    out_path = BASE_OUT_DIR / folder_clean
    out_path.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(
        [f for f in os.listdir(data_path) if f.lower().endswith(".vtk")],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )
    
    if not vtk_files:
        print(f"Skipping {folder_name}: No .vtk files found.")
        return

    sample_mesh = pv.read(str(data_path / vtk_files[0]))
    grid_size = int(np.sqrt(sample_mesh.n_points))
    
    print(f"Processing {folder_name} (Grid: {grid_size}x{grid_size})")

    cytokine_fields = np.zeros((N_TIMESTEPS, grid_size, grid_size, N_CYTOKINES), dtype=np.float32)
    cell_masks = np.zeros((N_TIMESTEPS, grid_size, grid_size, N_CELLS), dtype=np.float32)
    coords = None

    for i, file in enumerate(vtk_files[:N_TIMESTEPS]):
        mesh = pv.read(str(data_path / file))
        
        for j, ck in enumerate(CYTOKINE_NAMES):
            cytokine_fields[i, :, :, j] = mesh.point_data[ck].reshape(grid_size, grid_size, order="F")
        
        if 'CellType' in mesh.point_data:
            ct_data = mesh.point_data['CellType'].reshape(grid_size, grid_size, order="F")
            for j, (cell_name, cell_id) in enumerate(CELL_TYPES.items()):
                cell_masks[i, :, :, j] = (ct_data == cell_id).astype(np.float32)
        
        if coords is None:
            raw_coords = np.array(mesh.points).reshape(grid_size, grid_size, 3)[:, :, :2]
            coords = (raw_coords - raw_coords.min()) / (raw_coords.max() - raw_coords.min())

    # global scaling (0-1 normalization)
    scaler = MinMaxScaler()
    flat_fields = cytokine_fields.reshape(-1, N_CYTOKINES)
    scaled_fields = scaler.fit_transform(flat_fields).reshape(N_TIMESTEPS, grid_size, grid_size, N_CYTOKINES)
    
    # save scaler for inverse transform (essential for valid visualisations)
    dump(scaler, out_path / "scaler.joblib")
    
    # tensor generation 
    window = 2
    n_samples = N_TIMESTEPS - window
    t_norm = np.linspace(0, 1, N_TIMESTEPS)

    X_lstm = np.array([scaled_fields[i-window:i] for i in range(window, N_TIMESTEPS)])
    Y_target = np.array([scaled_fields[i] for i in range(window, N_TIMESTEPS)])

    X_branch = X_lstm.transpose(0, 2, 3, 1, 4).reshape(n_samples, grid_size, grid_size, -1)
    
    X_trunk = np.zeros((n_samples, grid_size * grid_size, 3))
    for s in range(n_samples):
        X_trunk[s, :, :2] = coords.reshape(-1, 2)
        X_trunk[s, :, 2] = t_norm[s + window]

    Y_masks = cell_masks[window:]

    np.save(out_path / "X_lstm.npy", X_lstm)
    np.save(out_path / "Y_target.npy", Y_target)
    np.save(out_path / "X_branch.npy", X_branch)
    np.save(out_path / "X_trunk.npy", X_trunk)
    np.save(out_path / "Y_masks.npy", Y_masks) 
    
    metadata = {
        "grid_size": grid_size, 
        "features": CYTOKINE_NAMES, 
        "cells": list(CELL_TYPES.keys()),
        "timesteps": N_TIMESTEPS
    }
    with open(out_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Success: Saved {grid_size}x{grid_size} tensors, masks, and scaler to {out_path}")

if __name__ == "__main__":
    if not BASE_LATTICE_DIR.exists():
        print(f"Error: {BASE_LATTICE_DIR} not found.")
    else:
        resolution_folders = [d for d in os.listdir(BASE_LATTICE_DIR) if os.path.isdir(BASE_LATTICE_DIR / d)]
        for folder in sorted(resolution_folders):
            try:
                process_mesh_resolution(folder)
            except Exception as e:
                print(f"Failed processing {folder}: {e}")
