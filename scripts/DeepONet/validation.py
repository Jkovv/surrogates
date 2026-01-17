import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim

def calculate_window_metrics(y_true, y_pred, grid_size):
    results = {}
    yt_f, yp_f = y_true.flatten(), y_pred.flatten()
    results["MSE"] = float(mean_squared_error(yt_f, yp_f))
    results["R2"] = float(r2_score(yt_f, yp_f))
    
    mask = y_true > 1e-8
    results["Masked_RMSE"] = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))) if np.any(mask) else 0.0

    ssim_vals = []
    for t in range(y_true.shape[0]):
        for c in range(6):
            d_range = max(1.0, np.max(y_true[t,:,:,c])) # Dynamiczny data_range
            s = ssim(y_true[t,:,:,c], y_pred[t,:,:,c], data_range=d_range)
            ssim_vals.append(s)
    results["SSIM"] = float(np.mean(ssim_vals))

    corrs = []
    for t in range(y_true.shape[0]):
        for c in range(6):
            gt, pr = y_true[t,:,:,c].flatten(), y_pred[t,:,:,c].flatten()
            if np.std(gt) > 1e-12:
                corrs.append(np.corrcoef(gt, pr)[0, 1])
    results["Spatial_Correlation"] = float(np.mean(corrs))
    return results
