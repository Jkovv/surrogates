import os, json, argparse, optuna
import numpy as np
from core_gpr import load_data_gpr
from validation_gpr import train_and_eval_gpr
from sklearn.metrics import mean_squared_error

def evaluate_windows(model, test_data):
    coords, Y_test_sims = test_data
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    results = {}
    
    grid_points = coords.shape[0] // 101
    for name, (start, end) in windows.items():
        mse_list = []
        for sim_idx in range(Y_test_sims.shape[0]):
            y_true = Y_test_sims[sim_idx, start:end].reshape(-1, 6)
            X_win = np.vstack([coords[t*grid_points : (t+1)*grid_points] 
                               for t in range(start, end)])
            y_pred = model.predict(X_win)
            mse_list.append(mean_squared_error(y_true, y_pred))
        results[name] = float(np.mean(mse_list))
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
    
    report = {
        "best_params": study.best_params,
        "window_results": evaluate_windows(final_model, test)
    }
    
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"GPR optimization complete for grid {args.grid}.")