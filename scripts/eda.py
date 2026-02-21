import os
import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

DATA_DIR = Path("./LatticeData")
OUTPUT_DIR = Path("./eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]

def extract_grid_size(name):
    match = re.search(r'(\d+)x(\d+)', name)
    if match: return int(match.group(1))
    return None

def run_deep_eda():
    all_stats = []
    folders = sorted([f for f in os.listdir(DATA_DIR) if (DATA_DIR / f).is_dir()])
    
    for folder in folders:
        case_path = DATA_DIR / folder
        grid = extract_grid_size(folder)
        if grid is None: continue
        
        print(f"\nAnalyzing Grid: {grid} ({folder})")
        vtk_files = sorted([f for f in os.listdir(case_path) if f.endswith(".vtk")])
        if not vtk_files: continue

        mid_idx = len(vtk_files) // 2
        mesh = pv.read(str(case_path / vtk_files[mid_idx]))
        
        for cyto in CYTOKINES:
            if cyto not in mesh.point_data: continue
            vals = mesh.point_data[cyto]
            max_val = np.max(vals)
            non_zeros = vals[vals > 0]
            
            all_stats.append({
                "Grid": grid,
                "Cytokine": cyto,
                "Raw_Max": max_val,
                "Raw_Min": np.min(vals),
                "Raw_Mean": np.mean(vals),
                "NonZero_Mean": np.mean(non_zeros) if non_zeros.size > 0 else 0,
                "Sparsity_Pct": (1.0 - (non_zeros.size / vals.size)) * 100,
                "Order_of_Mag": int(np.floor(np.log10(max_val))) if max_val > 0 else -20
            })

        # temporal corr
        sample_rate = 20 if grid >= 500 else 10
        print(f"Calculating temporal correlations (sampling every {sample_rate} steps)...")
        
        all_time_corrs = []
        
        for i in range(0, len(vtk_files), sample_rate):
            m = pv.read(str(case_path / vtk_files[i]))
            time_data = {c: m.point_data[c] for c in CYTOKINES if c in m.point_data}
            if time_data:
                all_time_corrs.append(pd.DataFrame(time_data).corr())

        # avg corr matrices over the whole timeline
        if all_time_corrs:
            avg_corr = pd.concat(all_time_corrs).groupby(level=0).mean()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_corr, annot=True, cmap="vlag", center=0, fmt=".2f")
            plt.title(f"Average Temporal Correlation Matrix ({grid}x{grid})")
            plt.savefig(OUTPUT_DIR / f"temporal_correlation_{grid}.png")
            plt.close()

        # dist plots 
        for cyto in ["il8", "tgf"]:
            vals = mesh.point_data[cyto]
            plt.figure(figsize=(8, 6))
            plt.hist(vals, bins=100, color='blue', alpha=0.7)
            plt.yscale('log')
            plt.title(f"Raw Distribution: {cyto.upper()} ({grid}x{grid})")
            plt.xlabel("Physical Concentration")
            plt.ylabel("Frequency (Log-Scale)")
            plt.savefig(OUTPUT_DIR / f"raw_hist_{grid}_{cyto}.png")
            plt.close()

    # final table 
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv(OUTPUT_DIR / "raw_data_analysis.csv", index=False)
    print(f"\nResults and plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_deep_eda()