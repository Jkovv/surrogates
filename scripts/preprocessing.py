import os
import numpy as np
import pyvista as pv
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# --- SET YOUR PATHS HERE ---
BASE_LATTICE_DIR = Path("./LatticeData")
BASE_OUT_DIR = Path("./preprocessed")

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
N_FEATURES = len(CYTOKINE_NAMES)

def process_mesh_resolution(folder_name):
    """
    Processes VTK files, generates scaled tensors, and saves the scaler 
    required to fix the 'fucked up' visualizations later.
    """
    res_path = BASE_LATTICE_DIR / folder_name
    out_path = BASE_OUT_DIR / folder_name
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing Resolution: {folder_name} ---")

    # 1. Sort VTK files numerically (Fixes temporal logic errors)
    vtk_files = sorted(
        [f for f in res_path.glob("*.vtk")], 
        key=lambda x: int(''.join(filter(str.isdigit, x.name)) or 0)
    )

    if not vtk_files:
        print(f"Skipping: No VTK files found in {res_path}")
        return

    # 2. Identify Grid Dimensions
    sample_mesh = pv.read(vtk_files[0])
    dims = sample_mesh.dimensions # (nx, ny, nz)
    grid_h, grid_w = dims[0], dims[1]
    print(f"Detected Grid: {grid_h}x{grid_w}")
    
    all_timesteps = []

    # 3. Extract Cytokine Fields
    for f in vtk_files:
        mesh = pv.read(f)
        step_data = np.zeros((grid_h, grid_w, N_FEATURES))
        
        for i, name in enumerate(CYTOKINE_NAMES):
            if name in mesh.point_data:
                # Reshape flat VTK data to 2D grid
                field = mesh.point_data[name].reshape(grid_h, grid_w, order="F")
                step_data[..., i] = field
            else:
                step_data[..., i] = 0.0
                
        all_timesteps.append(step_data)

    # Tensor Shape: (Timesteps, Height, Width, 6)
    full_tensor = np.array(all_timesteps, dtype=np.float32)

    # 4. Global Scaling & Scaler Persistence (Fixes Visualizations)
    # We flatten spatially to scale based on cytokine concentration values
    n_steps = full_tensor.shape[0]
    flat_fields = full_tensor.reshape(-1, N_FEATURES)
    
    scaler = MinMaxScaler()
    scaled_flat = scaler.fit_transform(flat_fields)
    
    # Reshape back to original 4D structure
    scaled_tensor = scaled_flat.reshape(n_steps, grid_h, grid_w, N_FEATURES)

    # 5. Save Results
    # 'X_branch' is the standard input for your DeepONet models
    np.save(out_path / "X_branch.npy", scaled_tensor)
    
    # CRITICAL: Save this scaler. You MUST load this in your plotting 
    # script to perform inverse_transform(), otherwise plots will stay broken.
    dump(scaler, out_path / "scaler.joblib")

    print(f"Success: Saved tensor {scaled_tensor.shape} and scaler to {out_path}")

if __name__ == "__main__":
    if not BASE_LATTICE_DIR.exists():
        print(f"FATAL ERROR: LatticeData directory not found at {BASE_LATTICE_DIR.absolute()}")
    else:
        # Get all subfolders (50x50, 100x100, etc.)
        folders = [d.name for d in BASE_LATTICE_DIR.iterdir() if d.is_dir()]
        
        for folder in sorted(folders):
            try:
                process_mesh_resolution(folder)
            except Exception as e:
                print(f"Error processing {folder}: {str(e)}")

    print("\nPreprocessing Complete. You can now train the DeepONet.")
