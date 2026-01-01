import os
import numpy as np

def load_data_gpr(grid_size, n_train_samples=3000):
    path = f"preprocessed/{grid_size}x{grid_size}"
    coords_raw = np.load(os.path.join(path, "X_trunk.npy")).astype(np.float32)
    data = np.load(os.path.join(path, "Y_target.npy")).astype(np.float32)
    num_timesteps = data.shape[0]
    
    coords = coords_raw[0] if coords_raw.ndim == 3 else coords_raw
    i_val, i_test = int(num_timesteps * 0.7), int(num_timesteps * 0.8)
    time_steps = np.arange(num_timesteps).astype(np.float32)

    X_list = [np.hstack([coords, np.full((coords.shape[0], 1), t)]) for t in time_steps]
    X_all, Y_all = np.vstack(X_list), data.reshape(-1, 6)

    n_points = coords.shape[0]
    train_end, val_end = i_val * n_points, i_test * n_points

    X_tr_p, Y_tr_p = X_all[:train_end], Y_all[:train_end]
    idx_tr = np.random.choice(len(X_tr_p), min(n_train_samples, len(X_tr_p)), replace=False)
    
    X_v_p, Y_v_p = X_all[train_end:val_end], Y_all[train_end:val_end]
    idx_v = np.random.choice(len(X_v_p), min(1000, len(X_v_p)), replace=False)

    return (X_tr_p[idx_tr], Y_tr_p[idx_tr]), \
           (X_v_p[idx_v], Y_v_p[idx_v]), \
           (coords, data, time_steps, i_test)
