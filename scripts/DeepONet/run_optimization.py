import os, json, argparse, optuna
import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from core import load_data_deeponet
from validation import train_and_eval

def calculate_dice(y_true, y_pred, percentile=90):
    thresh = np.percentile(y_true, percentile)
    mask_t, mask_p = y_true > thresh, y_pred > thresh
    intersection = np.logical_and(mask_t, mask_p).sum()
    return (2. * intersection) / (mask_t.sum() + mask_p.sum() + 1e-7)

def calculate_emd(y_true, y_pred):
    return wasserstein_distance(y_true.flatten(), y_pred.flatten())

def evaluate_windows_comprehensive(model, test_tuple, coords):
    X_b_test, Y_t_test = test_tuple
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    report = {}
    
    for name, (start, end) in windows.items():
        # practically from 82
        idx_s = max(0, start - 81) 
        idx_e = min(len(X_b_test), end - 81)
        
        if idx_s >= idx_e: continue
        
        preds = model.predict((X_b_test[idx_s:idx_e], coords))
        targets = Y_t_test[idx_s:idx_e]
        
        dice_list = [calculate_dice(targets[i], preds[i]) for i in range(len(targets))]
        emd_list = [calculate_emd(targets[i], preds[i]) for i in range(len(targets))]
        
        t_means = np.mean(targets, axis=(1, 2))
        p_means = np.mean(preds, axis=(1, 2))
        r2_traj = r2_score(t_means, p_means)
        
        report[name] = {
            "RMSE": float(np.sqrt(np.mean((preds - targets)**2))),
            "R2_Trajectory": float(r2_traj),
            "Dice": float(np.mean(dice_list)),
            "EMD": float(np.mean(emd_list))
        }
    return report

def objective(trial, train_raw, val_raw, b_dim, t_dim, coords):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 128, 256, step=64),
        'latent_dim': trial.suggest_int("latent_dim", 64, 128), 
        'lr': trial.suggest_float("lr", 1e-4, 5e-4, log=True),
        'activation': trial.suggest_categorical("activation", ["tanh", "relu"]),
        'epochs': 2000 
    }
    idx_p = np.random.choice(coords.shape[0], 2500, replace=False)
    train_data = (train_raw[0], coords[idx_p], train_raw[1][:, idx_p, :])
    val_data = (val_raw[0], coords[idx_p], val_raw[1][:, idx_p, :])
    
    train_state, _ = train_and_eval(params, train_data, val_data, b_dim, t_dim, seed=42)
    return train_state.best_metrics[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="250")
    args = parser.parse_args()
    grid_size = int(args.grid)
    
    train_raw, val_raw, test_raw, coords = load_data_deeponet(grid_size)
    b_dim, t_dim = train_raw[0].shape[1], coords.shape[1] 
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train_raw, val_raw, b_dim, t_dim, coords), n_trials=10)
    
    seeds = [1, 42, 100]
    best_p = study.best_params
    best_p['epochs'] = 5000 
    
    save_dir = f"models/deeponet_f/{grid_size}x{grid_size}"
    os.makedirs(save_dir, exist_ok=True)
    
    all_seed_results = []
    for s in seeds:
        idx_f = np.random.choice(coords.shape[0], min(10000, coords.shape[0]), replace=False)
        f_train = (train_raw[0], coords[idx_f], train_raw[1][:, idx_f, :])
        f_val = (val_raw[0], coords[idx_f], val_raw[1][:, idx_f, :])
        
        train_state, final_model = train_and_eval(best_p, f_train, f_val, b_dim, t_dim, seed=s)
        final_model.save(os.path.join(save_dir, f"model_seed_{s}"))
        
        win_metrics = evaluate_windows_comprehensive(final_model, test_raw, coords)
        
        all_seed_results.append({
            "seed": s,
            "train_mse": float(train_state.best_metrics[0]),
            "val_mse": float(train_state.best_metrics[0]), 
            "r2_score": float(train_state.best_metrics[1]), 
            "windows": win_metrics
        })
        
    report = {
        "best_params": best_p,
        "stability_summary": {
            "test_r2_avg": float(np.mean([r["r2_score"] for r in all_seed_results])),
            "test_r2_std": float(np.std([r["r2_score"] for r in all_seed_results])),
            "train_mse_avg": float(np.mean([r["train_mse"] for r in all_seed_results]))
        },
        "detailed_seeds": all_seed_results
    }
    
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"DeepONet optimization finished for grid {grid_size}")
