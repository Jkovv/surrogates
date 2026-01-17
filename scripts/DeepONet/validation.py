import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim

def calculate_window_metrics(y_true, y_pred, grid_size):
    results = {}
    yt_f, yp_f = y_true.flatten(), y_pred.flatten()
    
    results["MSE"] = float(mean_squared_error(yt_f, yp_f))
    results["MAE"] = float(mean_absolute_error(yt_f, yp_f))
    results["R2"] = float(r2_score(yt_f, yp_f))

    # masked RMSE 
    mask = y_true > 1e-8
    results["Masked_RMSE"] = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))) if np.any(mask) else 0.0

    # SSIM 
    ssim_list = []
    for t in range(y_true.shape[0]):
        for c in range(6):
            s = ssim(y_true[t,:,:,c], y_pred[t,:,:,c], data_range=1.0)
            ssim_list.append(s)
    results["SSIM"] = float(np.mean(ssim_list))

    # spatial correlation 
    corr_list = []
    for t in range(y_true.shape[0]):
        for c in range(6):
            gt, pr = y_true[t,:,:,c].flatten(), y_pred[t,:,:,c].flatten()
            if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
                corr_list.append(np.corrcoef(gt, pr)[0, 1])
            else:
                corr_list.append(0.0)
    results["Spatial_Correlation"] = float(np.mean(corr_list))
    return results
