import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim

def calculate_window_metrics(y_true, y_pred, grid_size):
    metrics = {}
    CYTOKINES = ["IL-8", "IL-1", "IL-6", "IL-10", "TNF", "TGF"]
    
    for i, name in enumerate(CYTOKINES):
        yt, yp = y_true[..., i].flatten(), y_pred[..., i].flatten()
        
        # per-cytokine
        r2 = float(r2_score(yt, yp))
        ssim_list = [ssim(y_true[t,:,:,i], y_pred[t,:,:,i], data_range=1.0, win_size=3) 
                     for t in range(y_true.shape[0])]
        
        mask = y_true[..., i] > 0.05
        rmse = float(np.sqrt(mean_squared_error(y_true[..., i][mask], y_pred[..., i][mask]))) if np.any(mask) else 0.0
        
        metrics[name] = {
            "R2": r2,
            "SSIM": float(np.mean(ssim_list)),
            "Masked_RMSE": rmse
        }
    
    # OVERALL
    ytf, ypf = y_true.flatten(), y_pred.flatten()
    r2_log = float(r2_score(ytf, ypf))
    mse_log = float(mean_squared_error(ytf, ypf))
    
    corrs = []
    for t in range(y_true.shape[0]):
        for c in range(6):
            c_val = np.corrcoef(y_true[t,:,:,c].flatten(), y_pred[t,:,:,c].flatten())[0, 1]
            if not np.isnan(c_val): corrs.append(c_val)

    metrics["OVERALL"] = {
        "MSE_log": mse_log,
        "R2_log": r2_log,
        "Masked_RMSE": float(np.sqrt(mean_squared_error(y_true[y_true > 0.05], y_pred[y_true > 0.05]))),
        "SSIM": float(np.mean([metrics[c]["SSIM"] for c in CYTOKINES])),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0,
        "R2": r2_log, 
        "MSE": mse_log  #
    }
    return metrics
