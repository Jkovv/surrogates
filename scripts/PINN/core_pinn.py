import numpy as np
import os

def get_scaled_physics(grid_size):
    nx = int(grid_size)
    true_size, s_mcs, h_mcs = 5, 60.0, 1/60.0 
    areaconv = true_size**2 / nx**2
    D_raw = np.array([2.09e-6, 3e-7, 8.49e-8, 1.45e-8, 4.07e-9, 2.6e-7])
    k_raw = np.array([0.2, 0.6, 0.5, 0.5, 0.5*0.225, 0.5*(1/25)])
    return (D_raw * s_mcs / areaconv).astype(np.float32), (k_raw * h_mcs).astype(np.float32)

def load_data_pinn(grid_size):
    path = f"preprocessed/{grid_size}x{grid_size}"
    X = np.load(os.path.join(path, "X_pinn.npy")).astype(np.float32)
    Y = np.load(os.path.join(path, "Y_pinn.npy")).astype(np.float32)

    X[:, 0:2] = X[:, 0:2] / grid_size # normalization [0-1]
    n = X.shape[0]

    # 70/10/20 split
    i_val, i_test = int(n * 0.7), int(n * 0.8)
    return (X[:i_val], Y[:i_val]), (X[i_val:i_test], Y[i_val:i_test]), (X[i_test:], Y[i_test:])
