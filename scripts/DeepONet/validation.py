import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def calculate_window_metrics(y_true, y_pred, grid_size):
    """
    1. Masked RMSE (Local Accuracy)
    2. SSIM (Structural/Morphological)
    3. Spatial Correlation (Linear Alignment)
    4. AUC (Physiological Cytokine Load)
    5. TLCC (Temporal Synchrony/Lag)
    6. Temporal R2 (Trajectory Variance)
    """
    metrics = {}
    CYTOKINES = ["IL-8", "IL-1", "IL-6", "IL-10", "TNF", "TGF"]
    
    # Time steps T, Height H, Width W, Channels C
    T, H, W, C = y_true.shape

    for i, name in enumerate(CYTOKINES):
        yt_flat, yp_flat = y_true[..., i].flatten(), y_pred[..., i].flatten()
        
        # masked RMSE (>0.05)
        mask = y_true[..., i] > 0.05
        m_rmse = float(np.sqrt(mean_squared_error(yt_flat[mask.flatten()], yp_flat[mask.flatten()]))) if np.any(mask) else 0.0

        # SSIM averaged over time
        ssim_val = np.mean([ssim(y_true[t,:,:,i], y_pred[t,:,:,i], data_range=1.0, win_size=3) 
                           for t in range(T)])

        # spatial corr averaged over time 
        spatial_corrs = []
        for t in range(T):
            r_val, _ = pearsonr(y_true[t,:,:,i].flatten(), y_pred[t,:,:,i].flatten())
            if not np.isnan(r_val): spatial_corrs.append(r_val)
        avg_spatial_corr = np.mean(spatial_corrs) if spatial_corrs else 0.0

        # AUC
        y_true_ts = np.mean(y_true[..., i], axis=(1, 2)) 
        y_pred_ts = np.mean(y_pred[..., i], axis=(1, 2))
        
        auc_true = np.trapz(y_true_ts, dx=1.0)
        auc_pred = np.trapz(y_pred_ts, dx=1.0)
        auc_error = float(np.abs(auc_true - auc_pred))

        # TLCC
        sync_corr, _ = pearsonr(y_true_ts, y_pred_ts)
        
        lags = np.arange(-5, 6) # shifts of +/- 5 hours
        tlcc_scores = []
        for k in lags:
            if k == 0:
                tlcc_scores.append(sync_corr)
            elif k > 0:
                tlcc_scores.append(np.corrcoef(y_true_ts[k:], y_pred_ts[:-k])[0, 1])
            else:
                tlcc_scores.append(np.corrcoef(y_true_ts[:k], y_pred_ts[-k:])[0, 1])
        
        max_lag = int(lags[np.argmax(tlcc_scores)])

        metrics[name] = {
            "Temporal_R2": float(r2_score(yt_flat, yp_flat)),
            "Masked_RMSE": m_rmse,
            "SSIM": float(ssim_val),
            "Spatial_Correlation": float(avg_spatial_corr),
            "AUC_Load_Error": auc_error,
            "TLCC_Synchrony": float(sync_corr),
            "Peak_Lag": max_lag
        }

    # overall
    metrics["OVERALL"] = {
        "MSE_Global": float(mean_squared_error(y_true.flatten(), y_pred.flatten())),
        "R2_Global": float(r2_score(y_true.flatten(), y_pred.flatten())),
        "Avg_Masked_RMSE": float(np.mean([metrics[c]["Masked_RMSE"] for c in CYTOKINES])),
        "Avg_SSIM": float(np.mean([metrics[c]["SSIM"] for c in CYTOKINES])),
        "Avg_Spatial_Corr": float(np.mean([metrics[c]["Spatial_Correlation"] for c in CYTOKINES])),
        "Avg_AUC_Error": float(np.mean([metrics[c]["AUC_Load_Error"] for c in CYTOKINES])),
        "Avg_TLCC_Sync": float(np.mean([metrics[c]["TLCC_Synchrony"] for c in CYTOKINES]))
    }
    
    return metrics
