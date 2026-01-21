import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def calculate_window_metrics(y_true, y_pred, grid_size):
    metrics = {}
    CYTOKINES = ["IL-8", "IL-1", "IL-6", "IL-10", "TNF", "TGF"]
    T, H, W, C = y_true.shape

    for i, name in enumerate(CYTOKINES):
        yt, yp = y_true[..., i], y_pred[..., i]
        yt_f, yp_f = yt.flatten(), yp.flatten()

        mask = yt > 0.05
        m_rmse = float(np.sqrt(mean_squared_error(yt_f[mask.flatten()], yp_f[mask.flatten()]))) if np.any(mask) else 0.0
        
        ssim_val = np.mean([ssim(yt[t], yp[t], data_range=1.0, win_size=3) for t in range(T)])
        spatial_corr = np.mean([pearsonr(yt[t].flatten(), yp[t].flatten())[0] for t in range(T)])

        yt_ts, yp_ts = np.mean(yt, axis=(1, 2)), np.mean(yp, axis=(1, 2))
        auc_err = float(np.abs(np.trapz(yt_ts) - np.trapz(yp_ts)))

        sync = float(pearsonr(yt_ts, yp_ts)[0])
        lags = np.arange(-5, 6)
        tlcc = []
        for k in lags:
            if k == 0: tlcc.append(sync)
            elif k > 0: tlcc.append(np.corrcoef(yt_ts[k:], yp_ts[:-k])[0, 1])
            else: tlcc.append(np.corrcoef(yt_ts[:k], yp_ts[-k:])[0, 1])
        peak_lag = int(lags[np.argmax(tlcc)])

        metrics[name] = {
            "Temporal_R2": float(r2_score(yt_f, yp_f)),
            "Masked_RMSE": m_rmse,
            "SSIM": float(ssim_val),
            "Spatial_Correlation": float(spatial_corr),
            "AUC_Load_Error": auc_err,
            "TLCC_Synchrony": sync,
            "Peak_Lag": peak_lag
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
