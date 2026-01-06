import os, json, argparse, optuna, gc
import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from core import load_data_pideeponet
from validation import train_and_eval

def calculate_spatial_metrics(y_true, y_pred):
    thresh = np.percentile(y_true, 90)
    mask_t, mask_p = y_true > thresh, y_pred > thresh
    dice = (2. * np.logical_and(mask_t, mask_p).sum()) / (mask_t.sum() + mask_p.sum() + 1e-7)
    emd = wasserstein_distance(y_true.flatten(), y_pred.flatten())
    return dice, emd

def evaluate_windows(model, test_tuple, coords):
    X_b_test, Y_t_test = test_tuple
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    report = {}
    
    for name, (start, end) in windows.items():
        idx_s, idx_e = max(0, start - 81), min(len(X_b_test), end - 81)
        if idx_s >= idx_e: continue
        
        preds = model.predict((X_b_test[idx_s:idx_e], coords))
        targets = Y_t_test[idx_s:idx_e]
        
        dice_l = [calculate_spatial_metrics(targets[i], preds[i])[0] for i in range(len(targets))]
        emd_l = [calculate_spatial_metrics(targets[i], preds[i])[1] for i in range(len(targets))]
        r2_traj = r2_score(np.mean(targets, axis=(1, 2)), np.mean(preds, axis=(1, 2)))
        
        report[name] = {
            "RMSE": float(np.sqrt(np.mean((preds - targets)**2))),
            "Dice": float(np.mean(dice_l)), 
            "EMD": float(np.mean(emd_l)),
            "R2_Trajectory": float(r2_traj)
        }
    return report

def objective(trial, grid_size, train_raw, val_raw, coords):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 128, 256, step=64),
        'latent_dim': trial.suggest_int("latent_dim", 64, 128),
        'lr': trial.suggest_float("lr", 1e-4, 5e-4, log=True),
        'activation': trial.suggest_categorical("activation", ["tanh", "relu"]),
        'epochs': 1000 
    }
    idx_p = np.random.choice(coords.shape[0], min(2500, coords.shape[0]), replace=False)
    
    train_data = (train_raw[0], coords[idx_p], train_raw[1][:, idx_p, :])
    val_data = (val_raw[0], coords[idx_p], val_raw[1][:, idx_p, :])
    
    train_state, _ = train_and_eval(params, train_data, val_data, seed=42)
    return float(np.sum(train_state.best_loss_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="100")
    args = parser.parse_args()
    grid_size = int(args.grid)
    
    train_raw, val_raw, test_raw, coords = load_data_pideeponet(grid_size)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, grid_size, train_raw, val_raw, coords), n_trials=10)
    
    best_p = study.best_params
    best_p['epochs'] = 5000 
    
    save_dir = f"models/pi_deeponet_dde/{grid_size}x{grid_size}"
    os.makedirs(save_dir, exist_ok=True)
    
    detailed_seeds = []
    for s in [1, 42, 100]:
        idx_f = np.random.choice(coords.shape[0], min(10000, coords.shape[0]), replace=False)
        f_train = (train_raw[0], coords[idx_f], train_raw[1][:, idx_f, :])
        f_val = (val_raw[0], coords[idx_f], val_raw[1][:, idx_f, :])
        
        train_state, final_model = train_and_eval(best_p, f_train, f_val, seed=s)       
        final_model.save(os.path.join(save_dir, f"model_seed_{s}"))
        
        win_metrics = evaluate_windows(final_model, test_raw, coords)
        
        detailed_seeds.append({
            "seed": s,
            "train_mse": float(np.sum(train_state.best_loss_train)),
            "val_mse": float(np.sum(train_state.best_loss_train)), 
            "windows": win_metrics
        })
    
    report = {
        "model": "pi_deeponet",
        "grid": grid_size,
        "best_params": best_p,
        "detailed_seeds": detailed_seeds
    }
    
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"PI-DeepONet optimization finished for grid {grid_size}")
