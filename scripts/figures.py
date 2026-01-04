import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(root_path):
    all_results = []
    models_dir = os.path.join(root_path, "models")
    
    model_map = {
        "gpr": "GPR",
        "sta_lstm": "STA-LSTM",
        "pinn": "PINN",
        "deeponet_dde": "DeepONet",
        "pi_deeponet_dde": "PI-DeepONet"
    }
    
    for folder, name in model_map.items():
        m_path = os.path.join(models_dir, folder)
        if not os.path.isdir(m_path): continue
            
        for grid_size in ["50x50", "100x100", "250x250", "500x500"]:
            json_path = os.path.join(m_path, grid_size, "research_report.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                for s_entry in data.get("detailed_seeds", []):
                    seed = s_entry.get("seed")
                    for win, metrics in s_entry.get("windows", {}).items():
                        all_results.append({
                            "Model": name,
                            "Grid": int(grid_size.split('x')[0]),
                            "Seed": seed,
                            "Window": win,
                            "RMSE": metrics.get("RMSE"),
                            "Dice": metrics.get("Dice"),
                            "EMD": metrics.get("EMD"),
                            "R2": metrics.get("R2_Trajectory")
                        })
    return pd.DataFrame(all_results)

def generate_output(df, root_path):
    output_dir = os.path.join(root_path, "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    model_order = ["GPR", "STA-LSTM", "PINN", "DeepONet", "PI-DeepONet"] # order 
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    
    for win in sorted(df['Window'].unique(), reverse=True):
        print(f"\nRESULTS FOR {win} (Mean ± Std)")
        summary = df[df['Window'] == win].groupby(["Model", "Grid"]).agg({
            "RMSE": ["mean", "std"],
            "Dice": ["mean", "std"],
            "R2": ["mean", "std"]
        }).round(6)
        print(summary.to_string())

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.6) # todo: adjust the colors to the chosen ones 

    for g_size in sorted(df['Grid'].unique()):
        mask = (df['Grid'] == g_size)
        if df[mask].empty: continue
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(
            data=df[mask], 
            x="Model", 
            y="RMSE", 
            hue="Window", 
            palette="viridis",
            errorbar="sd",
            capsize=.1
        )
        
        plt.title(f"RMSE Performance: Comparison of Temporal Windows (Grid {g_size}x{g_size})")
        plt.ylabel("RMSE (Log Scale)")
        plt.yscale('log')
        plt.legend(title="Evaluation Window", loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"rmse_windows_grid_{g_size}.png"), dpi=300)
        plt.close()
        print(f"Generated: rmse_windows_grid_{g_size}.png")

    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=df, 
        x="Grid", 
        y="Dice", 
        hue="Model", 
        style="Window", 
        markers=True, 
        markersize=12, 
        lw=3
    )
    
    plt.title("Morphological Fidelity (Dice) across Resolutions and Windows")
    plt.xlabel("Grid Resolution (pixels)")
    plt.ylabel("Dice Coefficient")
    plt.ylim(0, 1.05)
    plt.xticks([50, 100, 250, 500])
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dice_scalability_windows.png"), dpi=300)
    plt.close()
    print("Generated: dice_scalability_windows.png")

if __name__ == "__main__":
    root = "/gpfs/scratch1/shared/jkowalczuk/surrogates/burns"
    data = load_data(root)
    if not data.empty:
        generate_output(data, root)
    else:
        print("Error: No data found in research_report.json files.")
