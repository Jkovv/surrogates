import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def calculate_window_metrics(y_true, y_pred, grid_size):
    metrics = {}
    CYTOKINES = ["IL-8", "IL-1", "IL-6", "IL-10", "TNF", "TGF"]
    T, H, W, C = y_true.shape

    for i, name in enumerate(CYTOKINES):
        yt = y_true[..., i]
        yp = y_pred[..., i]
        
        yt_flat, yp_flat = yt.flatten(), yp.flatten()
        
        # masked RMSE (>0.05)
        mask = yt > 0.05
        if np.any(mask):
            m_rmse = float(np.sqrt(mean_squared_error(yt_flat[mask.flatten()], yp_flat[mask.flatten()])))
        else:
            m_rmse = float(np.sqrt(mean_squared_error(yt_flat, yp_flat)))

        # SSIM
        ssim_list = []
        for t in range(T):
            s_val = ssim(yt[t], yp[t], data_range=max(yt[t].max(), 1.0), win_size=3)
            ssim_list.append(s_val)
        ssim_val = np.mean(ssim_list)

        # spatial correlation
        spatial_corrs = []
        for t in range(T):
            if np.std(yt[t]) > 1e-9 and np.std(yp[t]) > 1e-9:
                r_val, _ = pearsonr(yt[t].flatten(), yp[t].flatten())
                if not np.isnan(r_val): spatial_corrs.append(r_val)
        avg_spatial_corr = np.mean(spatial_corrs) if spatial_corrs else 0.0

        # AUC 
        y_true_ts = np.mean(yt, axis=(1, 2)) 
        y_pred_ts = np.mean(yp, axis=(1, 2))
        auc_error = float(np.abs(np.trapz(y_true_ts) - np.trapz(y_pred_ts)))

        # TLCC 
        lags = np.arange(-5, 6)
        tlcc_scores = []
        
        if np.std(y_true_ts) > 1e-9 and np.std(y_pred_ts) > 1e-9:
            for k in lags:
                if k == 0:
                    c, _ = pearsonr(y_true_ts, y_pred_ts)
                    tlcc_scores.append(c)
                elif k > 0:
                    tlcc_scores.append(np.corrcoef(y_true_ts[k:], y_pred_ts[:-k])[0, 1])
                else:
                    tlcc_scores.append(np.corrcoef(y_true_ts[:k], y_pred_ts[-k:])[0, 1])
            
            tlcc_scores = [x if not np.isnan(x) else -1.0 for x in tlcc_scores]
            max_lag = int(lags[np.argmax(tlcc_scores)])
            sync_corr = float(tlcc_scores[len(lags)//2]) # lag 0
        else:
            max_lag = 0
            sync_corr = 0.0

        metrics[name] = {
            "Temporal_R2": float(r2_score(yt_flat, yp_flat)),
            "Masked_RMSE": m_rmse,
            "SSIM": float(ssim_val),
            "Spatial_Correlation": float(avg_spatial_corr),
            "AUC_Load_Error": auc_error,
            "TLCC_Synchrony": sync_corr,
            "Peak_Lag": max_lag
        }

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
