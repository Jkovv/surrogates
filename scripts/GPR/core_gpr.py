import os
import numpy as np

def load_data_gpr(grid_size, n_train_samples=3000):
    path = f"preprocessed/{grid_size}x{grid_size}"
    coords_raw = np.load(os.path.join(path, "X_trunk.npy")).astype(np.float32)
    data = np.load(os.path.join(path, "Y_target.npy")).astype(np.float32)

    num_timesteps = data.shape[0]
    
    if coords_raw.ndim == 3:
        coords = coords_raw[0]
    else:
        coords = coords_raw

    i_val = int(num_timesteps * 0.7)
    i_test = int(num_timesteps * 0.8)
    
    time_steps = np.arange(num_timesteps).astype(np.float32)

    X_list = []
    for t in time_steps:
        t_col = np.full((coords.shape[0], 1), t)
        X_list.append(np.hstack([coords, t_col]))
    
    X_all = np.vstack(X_list)
    Y_all = data.reshape(-1, 6)

    n_points = coords.shape[0]
    train_end_idx = i_val * n_points
    val_end_idx = i_test * n_points

    X_train_p = X_all[:train_end_idx]
    Y_train_p = Y_all[:train_end_idx]
    idx_tr = np.random.choice(len(X_train_p), min(n_train_samples, len(X_train_p)), replace=False)
    train_set = (X_train_p[idx_tr], Y_train_p[idx_tr])

    X_val_p = X_all[train_end_idx:val_end_idx]
    Y_val_p = Y_all[train_end_idx:val_end_idx]
    idx_v = np.random.choice(len(X_val_p), min(1000, len(X_val_p)), replace=False)
    val_set = (X_val_p[idx_v], Y_val_p[idx_v])

    test_set = (coords, data, time_steps, i_test)
    
    return train_set, val_set, test_set