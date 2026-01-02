import os, json, argparse, optuna, joblib
import numpy as np
from core_gpr import load_data_gpr
from validation_gpr import train_and_eval_gpr
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import wasserstein_distance

def calculate_spatial_metrics(y_true, y_pred):
    thresh = np.percentile(y_true, 90)
    mask_t, mask_p = y_true > thresh, y_pred > thresh
    dice = (2. * np.logical_and(mask_t, mask_p).sum()) / (mask_t.sum() + mask_p.sum() + 1e-7)
    emd = wasserstein_distance(y_true.flatten(), y_pred.flatten())
    return dice, emd

def evaluate_windows_comprehensive(model, test_data):
    coords, data_full, time_steps, i_test = test_data
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    report, n_points = {}, coords.shape[0]

    for name, (start, end) in windows.items():
        actual_start, actual_end = max(start, i_test), min(end, data_full.shape[0])
        if actual_start >= actual_end: continue

        w_preds, w_targets = [], []
        dice_l, emd_l = [], []

        for t_idx in range(actual_start, actual_end):
            X_step = np.hstack([coords, np.full((n_points, 1), time_steps[t_idx])])
            y_p = model.predict(X_step)
            y_t = data_full[t_idx].reshape(-1, 6)
            w_preds.append(y_p); w_targets.append(y_t)
            d, e = calculate_spatial_metrics(y_t, y_p)
            dice_l.append(d); emd_l.append(e)

        w_preds, w_targets = np.array(w_preds), np.array(w_targets)
        r2_traj = r2_score(np.mean(w_targets, axis=(1, 2)), np.mean(w_preds, axis=(1, 2)))
        report[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(w_targets.flatten(), w_preds.flatten()))),
            "Dice": float(np.mean(dice_l)), "EMD": float(np.mean(emd_l)),
            "R2_Trajectory": float(r2_traj)
        }
    return report

def objective(trial, train, val):
    params = {'length_scale': trial.suggest_float("length_scale", 0.1, 20.0, log=True),
              'nu': trial.suggest_categorical("nu", [0.5, 1.5, 2.5]),
              'alpha': trial.suggest_float("alpha", 1e-10, 1e-2, log=True)}
    mse, _ = train_and_eval_gpr(params, train, val, seed=42)
    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    train, val, test = load_data_gpr(args.grid)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, val), n_trials=10)

    save_dir = f"models/gpr/{args.grid}x{args.grid}"; os.makedirs(save_dir, exist_ok=True)
    seeds, all_seed_results = [1, 42, 100], []

    for s in seeds:
        _, model = train_and_eval_gpr(study.best_params, train, val, seed=s)
        joblib.dump(model, os.path.join(save_dir, f"model_seed_{s}.joblib"))
        all_seed_results.append({"seed": s, "windows": evaluate_windows_comprehensive(model, test)})

    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump({"model": "gpr", "best_params": study.best_params, "detailed_seeds": all_seed_results}, f, indent=4)
    
    print(f"GPR optimization and 3-seed training finished for grid {args.grid}x{args.grid}")
