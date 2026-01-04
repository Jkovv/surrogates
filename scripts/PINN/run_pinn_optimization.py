import os, json, argparse, optuna, gc
import numpy as np
import tensorflow as tf
import deepxde as dde
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score

from core_pinn import load_data_pinn
from validation_pinn import create_pinn_model

def calculate_spatial_metrics(y_true, y_pred):
    thresh = np.percentile(y_true, 90)
    mask_t, mask_p = y_true > thresh, y_pred > thresh
    dice = (2. * np.logical_and(mask_t, mask_p).sum()) / (mask_t.sum() + mask_p.sum() + 1e-7)
    emd = wasserstein_distance(y_true.flatten(), y_pred.flatten())
    return float(dice), float(emd)

def evaluate_pinn_windows(model, test_data, grid_size):
    X_test, Y_test = test_data
    times = np.unique(X_test[:, 2])
    
    is_scaled = np.max(times) <= 1.1
    windows_cfg = {
        "Window_82_100": (0.82, 1.01) if is_scaled else (82, 101),
        "Window_72_89": (0.72, 0.90) if is_scaled else (72, 90)
    }
    
    report = {}
    for name, (start, end) in windows_cfg.items():
        mask_w = (times >= start) & (times < end)
        w_times = times[mask_w]
        if len(w_times) == 0: continue
            
        rmse_l, dice_l, emd_l, p_means, t_means = [], [], [], [], []
        for t in w_times:
            idx = np.where(X_test[:, 2] == t)[0]
            t_X, t_Y = X_test[idx], Y_test[idx]
            t_pred = model.predict(t_X)
            
            rmse_l.append(np.sqrt(np.mean((t_pred - t_Y)**2)))
            d, e = calculate_spatial_metrics(t_Y, t_pred)
            dice_l.append(d); emd_l.append(e)
            p_means.append(np.mean(t_pred)); t_means.append(np.mean(t_Y))
            
        report[name] = {
            "RMSE": float(np.mean(rmse_l)),
            "Dice": float(np.mean(dice_l)),
            "EMD": float(np.mean(emd_l)),
            "R2_Trajectory": float(r2_score(t_means, p_means)) if len(t_means) > 1 else 0.0
        }
    return report

def objective(trial, grid, train, val):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 64, 128, step=32),
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'activation': trial.suggest_categorical("activation", ["tanh", "relu"])
    }
    tf.keras.backend.clear_session(); gc.collect()
    res = create_pinn_model(params, grid, train, val)
    model = res[0] if isinstance(res, tuple) else res
    model.compile("adam", lr=params['lr'])
    _, train_state = model.train(iterations=2000) 
    return float(np.sum(train_state.best_loss_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    
    train, val, test = load_data_pinn(args.grid)
    save_dir = f"models/pinn/{args.grid}x{args.grid}"; os.makedirs(save_dir, exist_ok=True)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args.grid, train, val), n_trials=5)
    
    best_p = study.best_params
    all_results = []
    
    for s in [1, 42, 100]:
        tf.keras.backend.clear_session(); gc.collect()
        dde.config.set_random_seed(s); tf.keras.utils.set_random_seed(s)
        
        res = create_pinn_model(best_p, args.grid, train, val)
        model = res[0] if isinstance(res, tuple) else res
        model.compile("adam", lr=best_p['lr'])
        
        _, train_state = model.train(iterations=10000)
        model.save(os.path.join(save_dir, f"model_seed_{s}"))
        
        t_mse = float(np.mean((model.predict(train[0]) - train[1])**2))
        v_mse = float(np.mean((model.predict(val[0]) - val[1])**2))
        
        all_results.append({
            "seed": s,
            "train_mse": t_mse,
            "val_mse": v_mse,
            "windows": evaluate_pinn_windows(model, test, args.grid)
        })
        
    report = {
        "model": "pinn",
        "grid": args.grid,
        "best_params": best_p,
        "detailed_seeds": all_results
    }
    
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("PINN model saved.")
