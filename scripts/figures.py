import os, json, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

CUSTOM_PALETTE = ["#ffd700", "#ffb14e", "#fa8775", "#ea5f94", "#cd34b5", "#9d02d7", "#0000ff"] # chosen previously

MODEL_COLORS = {
    "GPR": CUSTOM_PALETTE[0],
    "STA-LSTM": CUSTOM_PALETTE[1],
    "PINN": CUSTOM_PALETTE[3],
    "DeepONet": CUSTOM_PALETTE[5],
    "PI-DeepONet": CUSTOM_PALETTE[6]
}

WINDOW_COLORS = {
    "Window_72_89": CUSTOM_PALETTE[2],
    "Window_82_100": CUSTOM_PALETTE[6]
}

def load_data(root_path):
    all_results, hyperparams = [], []
    models_dir = os.path.join(root_path, "models")
    model_map = {
        "gpr": "GPR", "sta_lstm": "STA-LSTM", "pinn": "PINN",
        "deeponet_dde": "DeepONet", "pi_deeponet_dde": "PI-DeepONet"
    }
    
    for folder, name in model_map.items():
        m_path = os.path.join(models_dir, folder)
        if not os.path.isdir(m_path): continue
        for grid_size in ["50x50", "100x100", "250x250", "500x500"]:
            json_path = os.path.join(m_path, grid_size, "research_report.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                hp = data.get("best_params", {})
                hyperparams.append({
                    "Model": name, "Grid": grid_size, 
                    "Best Hyperparameters": ", ".join([f"{k}: {v}" for k, v in hp.items()])
                })
                
                for s_entry in data.get("detailed_seeds", []):
                    for win, m in s_entry.get("windows", {}).items():
                        all_results.append({
                            "Model": name, "Grid": int(grid_size.split('x')[0]),
                            "Seed": s_entry.get("seed"), "Window": win,
                            "RMSE": m.get("RMSE"), "Dice": m.get("Dice"),
                            "EMD": m.get("EMD"), "R2": m.get("R2_Trajectory")
                        })
    return pd.DataFrame(all_results), pd.DataFrame(hyperparams)


def print_final_report(df, df_hp):
    print("METHODOLOGY: OPTIMIZED HYPERPARAMETERS PER GRID")
    print(df_hp.sort_values(["Model", "Grid"]).to_string(index=False))
    
    print("SEED STABILITY (Mean, Std, CV%) ACROSS BOTH WINDOWS")
    stability = df.groupby(["Model", "Grid", "Window"])["RMSE"].agg(["mean", "std"])
    stability['CV_%'] = (stability['std'] / stability['mean']) * 100
    print(stability.sort_index(level=[0, 1, 2]).round(4).to_string())
    print("\Models with CV < 5% are considered scientifically robust.")

def plot_radar(df_grid, g_size, out):
    metrics = ["RMSE", "Dice", "R2", "EMD"]
    sub = df_grid[df_grid['Window'] == "Window_82_100"].groupby("Model").mean(numeric_only=True)[metrics]
    sub_norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)
    sub_norm['RMSE'], sub_norm['EMD'] = 1 - sub_norm['RMSE'], 1 - sub_norm['EMD']

    categories, N = list(sub_norm.columns), len(sub_norm.columns)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model in sub_norm.index:
        values = sub_norm.loc[model].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model, color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.08, color=MODEL_COLORS[model])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(f"Model Balance (Grid {g_size})")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(out, f"radar_grid_{g_size}.png"), dpi=300)
    plt.close()

def generate_all(root_path):
    out_dir = os.path.join(root_path, "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    df, df_hp = load_data(root_path)
    if df.empty: 
        print("No data found to visualize.")
        return
    
    print_final_report(df, df_hp)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    for g_size in sorted(df['Grid'].unique()):
        df_g = df[df['Grid'] == g_size]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_g, x="Model", y="RMSE", hue="Window", palette=WINDOW_COLORS, capsize=.1)
        plt.yscale('log')
        plt.title(f"Temporal Stability Comparison (Grid {g_size})")
        plt.savefig(os.path.join(out_dir, f"rmse_bars_grid_{g_size}.png"), dpi=300)
        plt.close()

        plot_radar(df_g, g_size, out_dir)
        
        # RMSE vs Dice
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_g[df_g['Window'] == "Window_82_100"], 
                        x="RMSE", y="Dice", hue="Model", palette=MODEL_COLORS, s=250)
        plt.title(f"Accuracy-Fidelity Gap (Grid {g_size})")
        plt.savefig(os.path.join(out_dir, f"scatter_grid_{g_size}.png"), dpi=300)
        plt.close()
            
    # global scalability
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df[df['Window']=="Window_82_100"], 
                 x="Grid", y="Dice", hue="Model", palette=MODEL_COLORS, 
                 markers=True, markersize=12, lw=4)
    plt.title("Dice Scalability across Grid Resolutions")
    plt.xticks([50, 100, 250, 500])
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(out_dir, "global_dice_trend.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {out_dir}")

if __name__ == "__main__":
    generate_all("/gpfs/scratch1/shared/jkowalczuk/surrogates/burns")
