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
    return dice, emd

def evaluate_pinn_windows(model, test_data):
    X_t, Y_t = test_data
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)} 
    report = {}
    y_pred_all = model.predict(X_t)
    
    for name, (start, end) in windows.items():
        mask = (X_t[:, 2] >= start) & (X_t[:, 2] < end)
        if not np.any(mask): continue
        w_preds, w_targets = y_pred_all[mask], Y_t[mask]
        
        dice_l, emd_l = [], []
        unique_ts = np.unique(X_t[mask, 2])
        for ts in unique_ts:
            t_mask = X_t[mask, 2] == ts
            d, e = calculate_spatial_metrics(w_targets[t_mask], w_preds[t_mask])
            dice_l.append(d); emd_l.append(e)
            
        report[name] = {
            "RMSE": float(np.sqrt(np.mean((w_preds - w_targets)**2))),
            "Dice": float(np.mean(dice_l)),
            "EMD": float(np.mean(emd_l)),
            "R2_Trajectory": float(r2_score(np.mean(w_targets, axis=0), np.mean(w_preds, axis=0)))
        }
    return report

def objective(trial, grid, train, val):
    params = {'hidden_size': trial.suggest_int("hidden_size", 128, 256, step=64),
              'lr': trial.suggest_float("lr", 1e-4, 5e-4, log=True),
              'activation': trial.suggest_categorical("activation", ["tanh", "relu"])}
    model, _, _ = create_pinn_model(params, grid, train, val)
    model.compile("adam", lr=params['lr'])
    _, train_state = model.train(iterations=2000)
    loss = np.sum(train_state.best_loss_test)
    tf.keras.backend.clear_session(); return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    train, val, test = load_data_pinn(args.grid)
    save_dir = f"models/pinn_dde/{args.grid}x{args.grid}"; os.makedirs(save_dir, exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args.grid, train, val), n_trials=5)

    seeds = [1, 42, 100] 
    all_results = []
    for s in seeds:
        dde.config.set_random_seed(s); tf.keras.utils.set_random_seed(s)
        model, D_v, k_v = create_pinn_model(study.best_params, args.grid, train, val)
        model.compile("adam", lr=study.best_params['lr'])
        model.train(iterations=8000)
        model.save(os.path.join(save_dir, f"model_seed_{s}"))
        
        all_results.append({
            "seed": s,
            "learned_physics": {"D": [float(dde.backend.to_numpy(v)) for v in D_v],
                                "k": [float(dde.backend.to_numpy(v)) for v in k_v]},
            "windows": evaluate_pinn_windows(model, test)
        })
        tf.keras.backend.clear_session(); gc.collect()

    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump({"model": "pinn", "grid": args.grid, "detailed_seeds": all_results}, f, indent=4)
