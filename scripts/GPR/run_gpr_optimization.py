import os, json, argparse, optuna, joblib
import numpy as np
from core_gpr import load_data_gpr
from validation_gpr import train_and_eval_gpr
from sklearn.metrics import mean_squared_error

def evaluate_windows(model, test_data):
    coords, data_full, time_steps, i_test = test_data
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    results = {}
    n_points = coords.shape[0]
    max_t = data_full.shape[0]

    for name, (start, end) in windows.items():
        actual_start = max(start, i_test)
        actual_end = min(end, max_t)
        
        if actual_start >= actual_end:
            results[name] = "Out of test range or data too short"
            continue

        y_true_win = data_full[actual_start:actual_end]
        y_true_flat = y_true_win.reshape(-1, 6)

        X_win_list = []
        for t_idx in range(actual_start, actual_end):
            t_col = np.full((n_points, 1), time_steps[t_idx])
            X_win_list.append(np.hstack([coords, t_col]))

        X_win = np.vstack(X_win_list)
        y_pred = model.predict(X_win)
        results[name] = float(mean_squared_error(y_true_flat, y_pred))
        
    return results

def objective(trial, train, val):
    params = {
        'length_scale': trial.suggest_float("length_scale", 0.1, 20.0, log=True),
        'nu': trial.suggest_categorical("nu", [0.5, 1.5, 2.5]),
        'alpha': trial.suggest_float("alpha", 1e-10, 1e-2, log=True)
    }
    mse, _ = train_and_eval_gpr(params, train, val)
    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="50")
    args = parser.parse_args()

    train, val, test = load_data_gpr(int(args.grid))

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, val), n_trials=10)

    _, final_model = train_and_eval_gpr(study.best_params, train, val)

    save_dir = f"models/gpr/{args.grid}x{args.grid}"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "gpr_model.joblib")
    joblib.dump(final_model, model_path)

    report = {
        "best_params": study.best_params,
        "model_file": "gpr_model.joblib",
        "window_results": evaluate_windows(final_model, test)
    }

    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"GPR complete. Model saved to {model_path}")