import os
import numpy as np

def load_data_gpr(grid_size, n_train_samples=3000):
    path = f"preprocessed/{grid_size}x{grid_size}"
    coords = np.load(os.path.join(path, "X_trunk.npy"))
    data = np.load(os.path.join(path, "Y_target.npy"))
    
    num_sim = data.shape[0]
    i_val, i_test = int(num_sim * 0.7), int(num_sim * 0.8)
    
    train_raw = data[:i_val].reshape(-1, 6)
    val_raw = data[i_val:i_test].reshape(-1, 6)
    
    X_all_coords = np.tile(coords, (i_val, 1))
    X_val_coords = np.tile(coords, (i_test - i_val, 1))
    
    # bc of complexity of O(N^3), we subsample training data
    idx_train = np.random.choice(len(train_raw), n_train_samples, replace=False)
    idx_val = np.random.choice(len(val_raw), 1000, replace=False)
    
    train_set = (X_all_coords[idx_train], train_raw[idx_train])
    val_set = (X_val_coords[idx_val], val_raw[idx_val])
    test_set = (coords, data[i_test:]) 
    
    return train_set, val_set, test_set