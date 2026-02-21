import os
import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from scipy.stats import kurtosis, skew

DATA_DIR = Path("./LatticeData")
OUTPUT_DIR = Path("./eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
TARGET_GRIDS = [50, 100, 250, 500]

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
        if grid not in TARGET_GRIDS: continue
        
        print(f"\nGrid: {grid}x{grid}")
        vtk_files = sorted([f for f in os.listdir(case_path) if f.endswith(".vtk")])
        if not vtk_files: continue

        ts_metrics = {c: {
            "max": [], "min": [], "mean": [], "std": [], 
            "skew": [], "kurt": [], "nz_pct": []
        } for c in CYTOKINES}
        
        cumulative_samples = {c: [] for c in CYTOKINES}
        all_time_corrs = []

        step = 10 if grid >= 500 else (5 if grid >= 250 else 1)
        
        for i in range(0, len(vtk_files), step):
            mesh = pv.read(str(case_path / vtk_files[i]))
            current_t_data = {}
            
            for cyto in CYTOKINES:
                if cyto not in mesh.point_data: continue
                vals = mesh.point_data[cyto]
                
                ts_metrics[cyto]["max"].append(np.max(vals))
                ts_metrics[cyto]["min"].append(np.min(vals))
                ts_metrics[cyto]["mean"].append(np.mean(vals))
                ts_metrics[cyto]["std"].append(np.std(vals))
                
                # if var > 0
                if np.std(vals) > 1e-18:
                    ts_metrics[cyto]["skew"].append(skew(vals))
                    ts_metrics[cyto]["kurt"].append(kurtosis(vals))
                
                ts_metrics[cyto]["nz_pct"].append(np.count_nonzero(vals > 0) / vals.size)
                cumulative_samples[cyto].extend(vals[::max(1, step*2)].tolist())
                current_t_data[cyto.upper()] = vals
            
            if current_t_data:
                all_time_corrs.append(pd.DataFrame(current_t_data).corr())

        for cyto in CYTOKINES:
            if not ts_metrics[cyto]["max"]: continue
            
            g_max = np.max(ts_metrics[cyto]["max"])
            g_min = np.min(ts_metrics[cyto]["min"])
            
            avg_skew = np.mean(ts_metrics[cyto]["skew"]) if ts_metrics[cyto]["skew"] else np.nan
            avg_kurt = np.mean(ts_metrics[cyto]["kurt"]) if ts_metrics[cyto]["kurt"] else np.nan

            all_stats.append({
                "Grid": grid,
                "Cytokine": cyto.upper(),
                "Global_Min": f"{g_min:.4e}",
                "Global_Max": f"{g_max:.4e}",
                "Time_Avg_Mean": f"{np.mean(ts_metrics[cyto]['mean']):.4e}",
                "Time_Avg_Std": f"{np.mean(ts_metrics[cyto]['std']):.4e}",
                "Time_Avg_Sparsity_Pct": round((1.0 - np.mean(ts_metrics[cyto]['nz_pct'])) * 100, 4),
                "Global_Skewness": round(avg_skew, 2) if not np.isnan(avg_skew) else "N/A",
                "Global_Kurtosis": round(avg_kurt, 2) if not np.isnan(avg_kurt) else "N/A",
                "Order_of_Mag": int(np.floor(np.log10(g_max))) if g_max > 0 else -20
            })

            plt.figure(figsize=(9, 6))
            plt.hist(cumulative_samples[cyto], bins=100, color='darkblue', alpha=0.7, log=True)
            plt.title(f"Global Distribution: {cyto.upper()} ({grid}x{grid})")
            plt.xlabel("Concentration Value [Physical Units]")
            plt.ylabel("Sample Count (Grid Point x Time Step) [Log-Scale]")
            plt.grid(True, which="both", ls="-", alpha=0.1)
            plt.savefig(OUTPUT_DIR / f"global_hist_{grid}_{cyto}.png", dpi=200)
            plt.close()

        if all_time_corrs:
            avg_corr = pd.concat(all_time_corrs).groupby(level=0).mean()
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_corr, annot=True, cmap=CORR_CMAP, center=0, fmt=".3f", 
                        vmin=-1, vmax=1, square=True)
            plt.title(f"Mean Correlation Matrix ({grid}x{grid})")
            plt.savefig(OUTPUT_DIR / f"global_correlation_{grid}.png", dpi=300)
            plt.close()

    if all_stats:
        pd.DataFrame(all_stats).to_csv(OUTPUT_DIR / "summary.csv", index=False)
        print(f"\nResults saved to {OUTPUT_DIR}/summary.csv")

if __name__ == "__main__":
    run_global_characterization_suite()
