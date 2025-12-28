import os
import numpy as np

def load_data_deeponet(grid_size):
    path = f"preprocessed/{grid_size}x{grid_size}"
    data = np.load(os.path.join(path, "Y_target.npy")) 
    coords = np.load(os.path.join(path, "X_trunk.npy"))
    
    X_b, Y_t = [], []
    # t, t+1 -> t+2
    for t in range(data.shape[1] - 2):
        X_b.append(data[:, t:t+2, :, :, :]) 
        Y_t.append(data[:, t+2, :, :, :])   
    
    X_b = np.array(X_b).transpose(1, 0, 2, 3, 4, 5).reshape(data.shape[0], 99, -1)
    Y_t = np.array(Y_t).transpose(1, 0, 2, 3, 4).reshape(data.shape[0], 99, -1, 6)

    # 70/10/20 split
    n = X_b.shape[0]
    i_val, i_test = int(n * 0.7), int(n * 0.8)
    
    def format_for_dde(xb_slice, yt_slice):
        num_sim, num_t, num_grid = xb_slice.shape[0], xb_slice.shape[1], coords.shape[0]
        xb_flat = np.repeat(xb_slice.reshape(-1, xb_slice.shape[-1]), num_grid, axis=0) # branch 
        xt_flat = np.tile(coords, (num_sim * num_t, 1)) # trunk
        y_flat = yt_slice.reshape(-1, 6)
        return (xb_flat, xt_flat), y_flat

    train_data = format_for_dde(X_b[:i_val], Y_t[:i_val])
    val_data = format_for_dde(X_b[i_val:i_test], Y_t[i_val:i_test])
    test_data = (X_b[i_test:], Y_t[i_test:]) 

    return train_data, val_data, test_data, coords