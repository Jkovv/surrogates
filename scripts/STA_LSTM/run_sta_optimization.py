import os, json, argparse, optuna, gc
import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from core_sta_lstm import load_data_sta, STALSTM
from validation_sta_lstm import train_and_eval_sta_lstm

def calculate_spatial_metrics(y_true, y_pred):
    thresh = np.percentile(y_true, 90)
    mask_t, mask_p = y_true > thresh, y_pred > thresh
    dice = (2. * np.logical_and(mask_t, mask_p).sum()) / (mask_t.sum() + mask_p.sum() + 1e-7)
    emd = wasserstein_distance(y_true.flatten(), y_pred.flatten())
    return dice, emd

def evaluate_sta_windows(model, test_set):
    X_test, Y_test = test_set
    preds = model.predict(X_test, batch_size=1, verbose=0)
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    report = {}
    
    for name, (start, end) in windows.items():
        idx_s, idx_e = max(0, start - 80), min(len(X_test), end - 80)
        if idx_s >= idx_e: continue
        
        w_preds, w_targets = preds[idx_s:idx_e], Y_test[idx_s:idx_e]
        dice_l, emd_l = [], []
        for i in range(len(w_targets)):
            d, e = calculate_spatial_metrics(w_targets[i], w_preds[i])
            dice_l.append(d); emd_l.append(e)
            
        report[name] = {
            "RMSE": float(np.sqrt(np.mean((w_preds - w_targets)**2))),
            "Dice": float(np.mean(dice_l)),
            "EMD": float(np.mean(emd_l)),
            "R2_Trajectory": float(r2_score(np.mean(w_targets, axis=(1, 2)), np.mean(w_preds, axis=(1, 2))))
        }
    return report

def objective(trial, train_set, val_set, grid_size):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 32, 128, step=32),
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'activation': trial.suggest_categorical("activation", ["ReLU", "SiLU"])
    }
    model = train_and_eval_sta_lstm(params, train_set, val_set, 42, grid_size)
    preds = model.predict(val_set[0], batch_size=1, verbose=0)
    loss = np.mean((val_set[1] - preds)**2)
    tf.keras.backend.clear_session(); gc.collect()
    return float(loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    
    save_dir = f"models/sta_lstm/{args.grid}x{args.grid}"; os.makedirs(save_dir, exist_ok=True)
    train, val, test = load_data_sta(args.grid)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, val, args.grid), n_trials=5)

    seeds = [1, 42, 100]
    all_results = []
    for s in seeds:
        model = train_and_eval_sta_lstm(study.best_params, train, val, s, args.grid)
        model.save_weights(os.path.join(save_dir, f"model_seed_{s}.h5"))
        all_results.append({
            "seed": s,
            "windows": evaluate_sta_windows(model, test)
        })
        tf.keras.backend.clear_session(); gc.collect()

    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump({"model": "sta-lstm", "grid": args.grid, "best_params": study.best_params, "detailed_seeds": all_results}, f, indent=4)
