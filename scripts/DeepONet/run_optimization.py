import os
import json
import argparse
import optuna
import numpy as np
import tensorflow as tf
from core import load_data_deeponet
from validation import train_and_eval

def evaluate_windows(model, test_tuple, coords):
    X_b_test, Y_t_test = test_tuple
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    results = {}
    for name, (start, end) in windows.items():
        idx_s, idx_e = max(0, start - 80), min(len(X_b_test), end - 80)
        if idx_s >= idx_e: continue
        mses = []
        for i in range(idx_s, idx_e):
            pred = model.predict((X_b_test[i:i+1], coords))
            mses.append(np.mean((pred - Y_t_test[i:i+1])**2))
        results[name] = float(np.mean(mses))
    return results

def objective(trial, train_raw, val_raw, b_dim, t_dim, coords):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 128, 256, step=64),
        'latent_dim': trial.suggest_int("latent_dim", 64, 128), 
        'lr': trial.suggest_float("lr", 1e-4, 5e-4, log=True),
        'activation': trial.suggest_categorical("activation", ["tanh", "relu"]),
        'epochs': 2000 
    }
    num_pts = 2500
    idx_p = np.random.choice(coords.shape[0], num_pts, replace=False)
    sampled_coords = coords[idx_p]
    
    train_data = (train_raw[0], sampled_coords, train_raw[1][:, idx_p, :])
    val_data = (val_raw[0], sampled_coords, val_raw[1][:, idx_p, :])
    
    loss, _ = train_and_eval(params, train_data, val_data, b_dim, t_dim, seed=42)
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="250")
    args = parser.parse_args()
    grid_size = int(args.grid)

    train_raw, val_raw, test_raw, coords = load_data_deeponet(grid_size)
    b_dim, t_dim = train_raw[0].shape[1], coords.shape[1] 

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train_raw, val_raw, b_dim, t_dim, coords), n_trials=10)

    best_p = study.best_params
    best_p['epochs'] = 5000 
    
    num_pts_final = min(10000, coords.shape[0]) 
    
    idx_f = np.random.choice(coords.shape[0], num_pts_final, replace=False)
    f_train = (train_raw[0], coords[idx_f], train_raw[1][:, idx_f, :])
    f_val = (val_raw[0], coords[idx_f], val_raw[1][:, idx_f, :])
    
    _, final_model = train_and_eval(best_p, f_train, f_val, b_dim, t_dim, seed=42)

    save_dir = f"models/deeponet_dde/{grid_size}x{grid_size}"
    os.makedirs(save_dir, exist_ok=True)
    final_model.save(os.path.join(save_dir, "deeponet_model")) 

    report = {
        "best_params": study.best_params,
        "window_results": evaluate_windows(final_model, test_raw, coords)
    }
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"DeepONet {grid_size}x{grid_size} completed successfully.")