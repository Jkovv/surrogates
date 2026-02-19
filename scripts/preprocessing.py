import os
import numpy as np
import pyvista as pv
import json
import re
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_LATTICE_DIR = Path("./LatticeData")
BASE_OUT_DIR = Path("./preprocessed")
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
CELL_TYPES = {"EC": 1, "NN": 2, "NA": 3, "M1": 4, "M2": 5}
N_TIMESTEPS = 101

def extract_grid_size(name):
    match = re.search(r'(\d+)x(\d+)', name)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', name)
    if match:
        return int(match.group(1))
    return None

def process_mesh(folder_name):
    data_path = BASE_LATTICE_DIR / folder_name
    grid_size = extract_grid_size(folder_name)
    
    if grid_size is None:
        print(f"Skipping {folder_name}: No grid size found.")
        return

    out_path = BASE_OUT_DIR / f"{grid_size}x{grid_size}"
    out_path.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted([f for f in os.listdir(data_path) if f.endswith(".vtk")],
                       key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))[:N_TIMESTEPS]

    if not vtk_files:
        return

    raw_cyt = np.zeros((N_TIMESTEPS, grid_size, grid_size, 6), dtype=np.float32)
    masks = np.zeros((N_TIMESTEPS, grid_size, grid_size, 5), dtype=np.float32)
    
    for i, f in enumerate(vtk_files):
        mesh = pv.read(str(data_path / f))
        for j, ck in enumerate(CYTOKINE_NAMES):
            raw_cyt[i, :, :, j] = mesh.point_data[ck].reshape(grid_size, grid_size, order="F")
        if 'CellType' in mesh.point_data:
            ct = mesh.point_data['CellType'].reshape(grid_size, grid_size, order="F")
            for j, (_, cid) in enumerate(CELL_TYPES.items()):
                masks[i, :, :, j] = (ct == cid).astype(np.float32)

    # raw data for PINN/Physics residuals
    np.save(out_path / "Y_raw_phys.npy", raw_cyt)

    # Log-scaling 
    log_data = np.log1p(raw_cyt * 1e7) 
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(log_data.reshape(-1, 6)).reshape(raw_cyt.shape)

    n_samples = N_TIMESTEPS - 2
    
    # LSTM/Branch input sequences (Window size 2)
    X_lstm = np.stack([scaled[i-2:i] for i in range(2, 101)])
    Y_target = scaled[2:]
    np.save(out_path / "X_lstm.npy", X_lstm)
    np.save(out_path / "Y_target.npy", Y_target)

    # Trunk input: spatio-temporal coordinates (x, y, t)
    coords = np.stack(np.meshgrid(np.linspace(0, 1, grid_size), 
                                 np.linspace(0, 1, grid_size), indexing='ij'), -1)
    coords_flat = coords.reshape(-1, 2)
    t_norm = np.linspace(0, 1, N_TIMESTEPS)[2:] 
    
    X_trunk = np.zeros((n_samples, grid_size * grid_size, 3), dtype=np.float32)
    for i in range(n_samples):
        X_trunk[i, :, :2] = coords_flat
        X_trunk[i, :, 2] = t_norm[i]
    
    np.save(out_path / "X_trunk.npy", X_trunk)

    # masks for evaluation and physics constraints
    Y_masks_spatial = masks[2:]
    np.save(out_path / "Y_masks_spatial.npy", Y_masks_spatial)
    np.save(out_path / "Y_masks_pinn.npy", Y_masks_spatial.reshape(n_samples, -1, 5))

    # scaling metadata for denormalization
    with open(out_path / "metadata.json", "w") as f:
        json.dump({
            "grid": grid_size, 
            "scaling": {"min": scaler.data_min_.tolist(), "max": scaler.data_max_.tolist()},
            "cytokines": CYTOKINE_NAMES
        }, f, indent=4)
    
    print(f"Processed: {grid_size}x{grid_size}")

if __name__ == "__main__":
    for folder in sorted(os.listdir(BASE_LATTICE_DIR)):
        if (BASE_LATTICE_DIR / folder).is_dir():
            process_mesh(folder)
