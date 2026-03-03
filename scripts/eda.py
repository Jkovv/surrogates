import os
import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

DATA_DIR = Path("./LatticeData")
OUTPUT_DIR = Path("./eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
TARGET_GRIDS = [50, 100, 250, 500]

def compute_morans_i(values, coords, n_neighbors=8):
    n = len(values)
    if n < 10 or np.std(values) < 1e-18:
        return np.nan

    if n > 10000:
        idx = np.random.choice(n, 10000, replace=False)
        values, coords = values[idx], coords[idx]
        n = 10000

    z = values - np.mean(values)
    z_sq_sum = np.sum(z ** 2)
    if z_sq_sum < 1e-30:
        return np.nan

    tree = cKDTree(coords)
    W_sum = 0.0
    lag_sum = 0.0
    for i in range(n):
        _, nbrs = tree.query(coords[i], k=n_neighbors + 1)
        nbrs = nbrs[1:]  # excluding self
        for j in nbrs:
            W_sum += 1.0
            lag_sum += z[i] * z[j]

    if W_sum < 1:
        return np.nan
    return float((n / W_sum) * (lag_sum / z_sq_sum))


def compute_variogram(values, coords, n_bins=15, max_frac=0.5):
    n = len(values)
    if n < 10 or np.std(values) < 1e-18:
        return np.array([]), np.array([])

    max_pts = 2000
    if n > max_pts:
        idx = np.random.choice(n, max_pts, replace=False)
        values, coords = values[idx], coords[idx]

    dists = pdist(coords)
    diffs = pdist(values.reshape(-1, 1)) ** 2 * 0.5

    max_dist = np.max(dists) * max_frac
    bins = np.linspace(0, max_dist, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    gamma = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    bin_idx = np.digitize(dists, bins) - 1
    for b in range(n_bins):
        mask = bin_idx == b
        if np.sum(mask) > 0:
            gamma[b] = np.mean(diffs[mask])
            counts[b] = np.sum(mask)

    valid = counts > 10
    return centers[valid], gamma[valid]

def compute_temporal_autocorrelation(fields, lags):
    T, n_pts = fields.shape
    max_pts = min(500, n_pts)
    pts_idx = np.random.choice(n_pts, max_pts, replace=False)
    sub = fields[:, pts_idx]

    results = {}
    for lag in lags:
        if lag >= T:
            results[lag] = np.nan
            continue
        x, y = sub[:T - lag], sub[lag:]
        corrs = []
        for p in range(max_pts):
            if np.std(x[:, p]) > 1e-18 and np.std(y[:, p]) > 1e-18:
                c = np.corrcoef(x[:, p], y[:, p])[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        results[lag] = float(np.mean(corrs)) if corrs else np.nan
    return results


def compute_temporal_cross_correlation(mean_ts, lags):
    names = list(mean_ts.keys())
    n = len(names)
    results = {}
    for lag in lags:
        mat = np.full((n, n), np.nan)
        for i, c1 in enumerate(names):
            s1 = mean_ts[c1]
            T = len(s1)
            if lag >= T:
                continue
            for j, c2 in enumerate(names):
                s2 = mean_ts[c2]
                x, y = s1[:T - lag], s2[lag:]
                if np.std(x) > 1e-18 and np.std(y) > 1e-18:
                    mat[i, j] = np.corrcoef(x, y)[0, 1]
        results[lag] = pd.DataFrame(mat, index=names, columns=names)
    return results

def extract_grid_size(name):
    match = re.search(r'(\d+)x(\d+)', name)
    if match: return int(match.group(1))
    match = re.search(r'(\d+)', name)
    if match: return int(match.group(1))
    return None

def run_global_characterization_suite():
    all_stats = []
    folders = sorted([f for f in os.listdir(DATA_DIR) if (DATA_DIR / f).is_dir()])
    CORR_CMAP = "coolwarm"

    for folder in folders:
        case_path = DATA_DIR / folder
        grid = extract_grid_size(folder)
        if grid not in TARGET_GRIDS:
            continue

        print(f"\nGrid: {grid}x{grid}")
        vtk_files = sorted([f for f in os.listdir(case_path) if f.endswith(".vtk")])
        if not vtk_files:
            continue

        ts_metrics = {c: {
            "max": [], "min": [], "mean": [], "std": [],
            "skew": [], "kurt": [], "nz_pct": []
        } for c in CYTOKINES}

        cumulative_samples = {c: [] for c in CYTOKINES}
        all_time_corrs = []

        coords = None
        timestep_indices = []
        spatial_fields = {c: [] for c in CYTOKINES}   # will become (T, n_pts)
        mean_time_series = {c: [] for c in CYTOKINES}  # (T,)

        step = 10 if grid >= 500 else (5 if grid >= 250 else 1)

        for i in range(0, len(vtk_files), step):
            mesh = pv.read(str(case_path / vtk_files[i]))
            current_t_data = {}

            if coords is None:
                coords = mesh.points[:, :2].astype(np.float64)

            timestep_indices.append(i)

            for cyto in CYTOKINES:
                if cyto not in mesh.point_data:
                    continue
                vals = mesh.point_data[cyto]

                ts_metrics[cyto]["max"].append(np.max(vals))
                ts_metrics[cyto]["min"].append(np.min(vals))
                ts_metrics[cyto]["mean"].append(np.mean(vals))
                ts_metrics[cyto]["std"].append(np.std(vals))

                # if var > 0
                if np.std(vals) > 1e-18:
                    ts_metrics[cyto]["skew"].append(skew(vals))
                    ts_metrics[cyto]["kurt"].append(kurtosis(vals))

                ts_metrics[cyto]["nz_pct"].append(
                    np.count_nonzero(vals > 0) / vals.size)
                cumulative_samples[cyto].extend(vals[::max(1, step * 2)].tolist())
                current_t_data[cyto.upper()] = vals

                spatial_fields[cyto].append(vals.copy())
                mean_time_series[cyto].append(np.mean(vals))

            if current_t_data:
                all_time_corrs.append(pd.DataFrame(current_t_data).corr())

        for cyto in CYTOKINES:
            mean_time_series[cyto] = np.array(mean_time_series[cyto])
            if spatial_fields[cyto]:
                spatial_fields[cyto] = np.stack(spatial_fields[cyto], axis=0)
            else:
                spatial_fields[cyto] = np.array([])

        T_actual = len(timestep_indices)
        print(f"  {T_actual} timesteps loaded (step={step})")

        for cyto in CYTOKINES:
            if not ts_metrics[cyto]["max"]:
                continue

            g_max = np.max(ts_metrics[cyto]["max"])
            g_min = np.min(ts_metrics[cyto]["min"])
            avg_skew = (np.mean(ts_metrics[cyto]["skew"])
                        if ts_metrics[cyto]["skew"] else np.nan)
            avg_kurt = (np.mean(ts_metrics[cyto]["kurt"])
                        if ts_metrics[cyto]["kurt"] else np.nan)

            all_stats.append({
                "Grid": grid,
                "Cytokine": cyto.upper(),
                "Global_Min": f"{g_min:.4e}",
                "Global_Max": f"{g_max:.4e}",
                "Time_Avg_Mean": f"{np.mean(ts_metrics[cyto]['mean']):.4e}",
                "Time_Avg_Std": f"{np.mean(ts_metrics[cyto]['std']):.4e}",
                "Time_Avg_Sparsity_Pct": round(
                    (1.0 - np.mean(ts_metrics[cyto]['nz_pct'])) * 100, 4),
                "Global_Skewness": (round(avg_skew, 2)
                                    if not np.isnan(avg_skew) else "N/A"),
                "Global_Kurtosis": (round(avg_kurt, 2)
                                    if not np.isnan(avg_kurt) else "N/A"),
                "Order_of_Mag": (int(np.floor(np.log10(g_max)))
                                 if g_max > 0 else -20)
            })

            plt.figure(figsize=(9, 6))
            plt.hist(cumulative_samples[cyto], bins=100,
                     color='darkblue', alpha=0.7, log=True)
            plt.title(f"Global Distribution: {cyto.upper()} ({grid}x{grid})")
            plt.xlabel("Concentration Value [Physical Units]")
            plt.ylabel("Sample Count (Grid Point x Time Step) [Log-Scale]")
            plt.grid(True, which="both", ls="-", alpha=0.1)
            plt.savefig(OUTPUT_DIR / f"global_hist_{grid}_{cyto}.png", dpi=200)
            plt.close()

        if all_time_corrs:
            avg_corr = pd.concat(all_time_corrs).groupby(level=0).mean()
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_corr, annot=True, cmap=CORR_CMAP, center=0,
                        fmt=".3f", vmin=-1, vmax=1, square=True)
            plt.title(f"Mean Correlation Matrix ({grid}x{grid})")
            plt.savefig(OUTPUT_DIR / f"global_correlation_{grid}.png", dpi=300)
            plt.close()

        print("  Spatial: Moran's I ...")
        sample_t = np.linspace(0, T_actual - 1, min(20, T_actual)).astype(int)

        morans = {c: [] for c in CYTOKINES}
        for cyto in CYTOKINES:
            if spatial_fields[cyto].size == 0:
                continue
            for ti in sample_t:
                morans[cyto].append(
                    compute_morans_i(spatial_fields[cyto][ti], coords))

        fig, ax = plt.subplots(figsize=(10, 5))
        for cyto in CYTOKINES:
            if morans[cyto]:
                ax.plot(sample_t, morans[cyto], 'o-',
                        label=cyto.upper(), markersize=4)
        ax.set_xlabel("Timestep index")
        ax.set_ylabel("Moran's I")
        ax.set_title(f"Spatial Autocorrelation per Cytokine ({grid}x{grid})")
        ax.legend()
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"spatial_morans_i_{grid}.png", dpi=200)
        plt.close()

        print("  Spatial: variograms ...")
        vt = [0, T_actual // 2, T_actual - 1]
        vlabels = ["Early", "Mid", "Late"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        for ci, cyto in enumerate(CYTOKINES):
            ax = axes.flat[ci]
            if spatial_fields[cyto].size == 0:
                continue
            for ti, lab in zip(vt, vlabels):
                h, gamma = compute_variogram(spatial_fields[cyto][ti], coords)
                if len(h) > 0:
                    ax.plot(h, gamma, 'o-', label=lab, markersize=3)
            ax.set_title(cyto.upper())
            ax.set_xlabel("Distance")
            ax.set_ylabel("Semi-variance")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Empirical Variograms ({grid}x{grid})", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"spatial_variograms_{grid}.png", dpi=200)
        plt.close()

        print("  Temporal: autocorrelation ...")
        max_lag = min(30, T_actual // 2)
        lags = list(range(1, max_lag + 1))

        fig, ax = plt.subplots(figsize=(10, 5))
        for cyto in CYTOKINES:
            if spatial_fields[cyto].size == 0:
                continue
            acorr = compute_temporal_autocorrelation(spatial_fields[cyto], lags)
            ax.plot(lags, [acorr[l] for l in lags], 'o-',
                    label=cyto.upper(), markersize=3)
        ax.set_xlabel("Lag (timesteps)")
        ax.set_ylabel("Mean Temporal Autocorrelation")
        ax.set_title(f"Temporal Autocorrelation per Cytokine ({grid}x{grid})")
        ax.legend()
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"temporal_autocorrelation_{grid}.png", dpi=200)
        plt.close()
        
        print("  Temporal: cross-correlation ...")
        cross_lags = [l for l in [0, 1, 5, 10] if l < T_actual // 2]
        mean_ts = {c.upper(): mean_time_series[c]
                   for c in CYTOKINES if len(mean_time_series[c]) > 0}

        if mean_ts and cross_lags:
            cc = compute_temporal_cross_correlation(mean_ts, cross_lags)
            fig, axes_cc = plt.subplots(1, len(cross_lags),
                                        figsize=(5 * len(cross_lags), 5))
            if len(cross_lags) == 1:
                axes_cc = [axes_cc]
            for ax, lag in zip(axes_cc, cross_lags):
                sns.heatmap(cc[lag], annot=True, cmap=CORR_CMAP, center=0,
                            fmt=".2f", vmin=-1, vmax=1, square=True, ax=ax,
                            cbar=(lag == cross_lags[-1]))
                ax.set_title(f"Lag = {lag}")
            fig.suptitle(
                f"Temporal Cross-Correlation ({grid}x{grid})", fontsize=14)
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"temporal_cross_correlation_{grid}.png", dpi=200)
            plt.close()

        fig, axes_ts = plt.subplots(2, 3, figsize=(15, 9))
        for ci, cyto in enumerate(CYTOKINES):
            ax = axes_ts.flat[ci]
            ts_mean = mean_time_series[cyto]
            if len(ts_mean) == 0:
                continue
            ts_std_arr = np.array(ts_metrics[cyto]["std"][:T_actual])
            ts_mean_arr = np.array(ts_metrics[cyto]["mean"][:T_actual])
            ax.plot(timestep_indices, ts_mean_arr, 'b-', linewidth=1)
            ax.fill_between(timestep_indices,
                            ts_mean_arr - ts_std_arr,
                            ts_mean_arr + ts_std_arr,
                            alpha=0.2, color='blue')
            ax.set_title(cyto.upper())
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Mean +/- Std")
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"Mean Cytokine Concentration Over Time ({grid}x{grid})",
            fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"temporal_mean_timeseries_{grid}.png", dpi=200)
        plt.close()

        print(f"  Done — plots saved to {OUTPUT_DIR}/")

    if all_stats:
        pd.DataFrame(all_stats).to_csv(OUTPUT_DIR / "summary.csv", index=False)
        print(f"\nAll results -> {OUTPUT_DIR}/summary.csv")


if __name__ == "__main__":
    run_global_characterization_suite()
