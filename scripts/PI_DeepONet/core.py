import os
import numpy as np

def get_scaled_physics(grid_size):
    nx = int(grid_size)
    true_size, s_mcs, h_mcs = 5, 60.0, 1/60.0
    areaconv = true_size**2 / nx**2
    D_raw = np.array([2.09e-6, 3e-7, 8.49e-8, 1.45e-8, 4.07e-9, 2.6e-7])
    k_raw = np.array([0.2, 0.6, 0.5, 0.5, 0.5*0.225, 0.5*(1/25)])
    return (D_raw * s_mcs / areaconv).astype(np.float32), (k_raw * h_mcs).astype(np.float32)

def rescale_branch_input(data, target_res=50):
    n, h, w, c = data.shape
    if h == target_res: return data
    factor = h // target_res
    reshaped = data[:, :target_res*factor, :target_res*factor, :]
    reshaped = reshaped.reshape(n, target_res, factor, target_res, factor, c)
    return reshaped.mean(axis=(2, 4))

def load_data_pideeponet(grid_size):
    path = f"preprocessed/{grid_size}x{grid_size}"
    raw_data = np.load(os.path.join(path, "X_branch.npy")).astype(np.float32)
    data = raw_data[..., :6]
    branch_sensors = rescale_branch_input(data, target_res=50)
    n_samples = len(data) - 2
    X_b = np.zeros((n_samples, 2, 50, 50, 6), dtype=np.float32)
    for i in range(n_samples):
        X_b[i] = branch_sensors[i:i+2]
    Y_t = data[2:]
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    gx, gy = np.meshgrid(x, y)
    coords = np.column_stack((gx.ravel(), gy.ravel())).astype(np.float32)
    X_b_flat = X_b.reshape(n_samples, -1)
    Y_t_3d = Y_t.reshape(n_samples, -1, 6)
    idx_val, idx_test = int(n_samples * 0.7), int(n_samples * 0.8)
    return (X_b_flat[:idx_val], Y_t_3d[:idx_val]), \
           (X_b_flat[idx_val:idx_test], Y_t_3d[idx_val:idx_test]), \
           (X_b_flat[idx_test:], Y_t_3d[idx_test:]), coords