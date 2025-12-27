import os
import json
import numpy as np
import optuna
import argparse
import tensorflow as tf
from core_sta_lstm import load_data_sta, STALSTM
from validation_sta_lstm import train_and_eval_sta_lstm

def run_window_evaluation(model, X_test, Y_test):
    """
    evaluating the model on specific research-relevant temporal windows - like before:
    - Window A: t=82 to t=100
    - Window B: t=72 to t=89
    """
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    preds = model.predict(X_test)
    results = {}
    
    for name, (start, end) in windows.items():
        idx_s, idx_e = start - 80, end - 80 
        mse = np.mean((Y_test[idx_s:idx_e] - preds[idx_s:idx_e])**2)
        results[name] = float(mse)
    return results

def objective(trial, train_set, val_set):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 64, 512, step=64),
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'activation': trial.suggest_categorical("activation", ["ReLU", "Tanh", "SiLU", "GELU"])
    }
    # Stability test - seeds 1, 42, and 100
    seeds = [1, 42, 100]
    seed_losses = [train_and_eval_sta_lstm(params, train_set, val_set, s)[0] for s in seeds]
    return np.mean(seed_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="50")
    args = parser.parse_args()

    # todo: maybe change later?
    grid_configs = {"50": 30, "100": 20, "250": 15, "500": 10}
    n_trials = grid_configs.get(args.grid, 10)
    
    save_dir = f"models/sta_lstm/{args.grid}x{args.grid}"
    os.makedirs(save_dir, exist_ok=True)

    # 70/10/20
    train, val, test = load_data_sta(args.grid)

    # hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, val), n_trials=n_trials)

    # final model with best params on seed 42
    best_p = study.best_params
    _, best_weights = train_and_eval_sta_lstm(best_p, train, val, seed=42)
    
    final_model = STALSTM(best_p['hidden_size'], train[1].shape[1:], act_name=best_p['activation'])
    final_model.set_weights(best_weights)

    # report 
    report = run_window_evaluation(final_model, test[0], test[1])
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump({"best_params": best_p, "results": report}, f, indent=4)
    
    print(f"STA-LSTM optimization for {args.grid}x{args.grid} completed.")