import os
import json
import numpy as np
import optuna
import argparse
import tensorflow as tf
import gc
from core_sta_lstm import load_data_sta, STALSTM
from validation_sta_lstm import train_and_eval_sta_lstm

def objective(trial, train_set, val_set, grid_size):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 32, 128, step=32),
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'activation': trial.suggest_categorical("activation", ["ReLU", "SiLU"])
    }
    seeds = [1, 42, 100]
    seed_losses = []
    for s in seeds:
        loss, _ = train_and_eval_sta_lstm(params, train_set, val_set, s, grid_size)
        seed_losses.append(loss)
        tf.keras.backend.clear_session()
        gc.collect()
    return np.mean(seed_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    
    save_dir = f"models/sta_lstm/{args.grid}x{args.grid}"
    os.makedirs(save_dir, exist_ok=True)

    train, val, test = load_data_sta(args.grid)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, val, args.grid), n_trials=5)

    best_p = study.best_params
    _, best_weights = train_and_eval_sta_lstm(best_p, train, val, 42, args.grid)
    
    final_model = STALSTM(best_p['hidden_size'], train[1].shape[1:], args.grid, best_p['activation'])
    final_model(np.zeros((1, 2, args.grid, args.grid, 6), dtype=np.float32))
    final_model.set_weights(best_weights)

    preds = final_model.predict(test[0], batch_size=1)
    results = {}
    for name, (s, e) in {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}.items():
        results[name] = float(np.mean((test[1][s-80:e-80] - preds[s-80:e-80])**2))

    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump({"best_params": best_p, "results": results}, f, indent=4)
    print(f"STA-LSTM {args.grid}x{args.grid} finished.")