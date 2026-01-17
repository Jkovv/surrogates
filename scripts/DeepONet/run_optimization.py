import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import json
import argparse
import numpy as np
import tensorflow as tf
import optuna
from pathlib import Path
from optuna_integration import TFKerasPruningCallback
from sklearn.metrics import r2_score

from core import build_deeponet
from validation import calculate_window_metrics

BASE_DATA_DIR = Path("./preprocessed")
MODEL_OUT_DIR = Path("./models/deeponet")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

def weighted_mse(y_true, y_pred):
    weight = tf.where(tf.greater(y_true, 1e-5), 10.0, 1.0)
    return tf.reduce_mean(tf.square(y_true - y_pred) * weight)

def prepare_flat_data(X_b, X_t, Y, num_samples=5000):
    steps = X_b.shape[0]
    num_total_points = X_t.shape[1]
    actual_samples = min(num_samples, num_total_points)
    indices = np.random.choice(num_total_points, actual_samples, replace=False)
    X_b_flat = np.repeat(X_b, actual_samples, axis=0)
    X_t_flat = X_t[:, indices, :].reshape(-1, 3)
    Y_flat = Y.reshape(steps, -1, 6)[:, indices, :].reshape(-1, 6)
    return X_b_flat, X_t_flat, Y_flat

def run_study(grid, n_trials=15):
    print(f"\nStarting loss-based optimization for grid: {grid}x{grid}")
    path = BASE_DATA_DIR / f"{grid}x{grid}"
    X_b_raw, X_t_raw, Y_raw = np.load(path/"X_branch.npy"), np.load(path/"X_trunk.npy"), np.load(path/"Y_target.npy")
    train_end, val_end = int(0.7 * len(X_b_raw)), int(0.8 * len(X_b_raw))

    def objective(trial):
        tf.keras.backend.clear_session()
        model = build_deeponet(trial, grid_size=grid)
        lr = trial.suggest_float("lr", 1e-5, 1.5e-4, log=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, global_clipnorm=0.5)
        model.compile(optimizer=optimizer, loss=weighted_mse)
        
        X_bt, X_tt, Yt = prepare_flat_data(X_b_raw[:train_end], X_t_raw[:train_end], Y_raw[:train_end], 5000)
        X_bv, X_tv, Yv = prepare_flat_data(X_b_raw[train_end:val_end], X_t_raw[train_end:val_end], Y_raw[train_end:val_end], 5000)

        model.fit([X_bt, X_tt], Yt, validation_data=([X_bv, X_tv], Yv),
                  epochs=100, batch_size=256, verbose=0,
                  callbacks=[TFKerasPruningCallback(trial, "val_loss")])
        
        val_loss = model.evaluate([X_bv, X_tv], Yv, verbose=0)
        return val_loss

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"\nBest params found: {best_params}")
    
    # final training
    model = build_deeponet(best_params, grid)
    model.compile(optimizer=tf.keras.optimizers.Adam(best_params["lr"]), loss=weighted_mse)
    X_bf, X_tf, Yf = prepare_flat_data(X_b_raw[:train_end], X_t_raw[:train_end], Y_raw[:train_end], grid*grid)
    
    print("\nFinal prod training (150 epochs)...")
    model.fit([X_bf, X_tf], Yf, epochs=150, batch_size=128, verbose=1)
    
    # saving weights
    weights_path = MODEL_OUT_DIR / f"weights_deeponet_{grid}_tuned.weights.h5"
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")
    
    # val on windows (slightly changed)
    windows = {"Window_82_100": (80, 99), "Window_72_89": (70, 88)}
    seed_data = {"seed": 42, "windows": {}}
    for name, (s, e) in windows.items():
        preds_grid = []
        for t in range(s, e):
            b_in = np.repeat(X_b_raw[t:t+1], grid*grid, axis=0)
            p = model.predict([b_in, X_t_raw[t]], verbose=0).reshape(grid, grid, 6)
            preds_grid.append(p)
        seed_data["windows"][name] = calculate_window_metrics(Y_raw[s:e], np.array(preds_grid), grid)
    
    # json
    with open(MODEL_OUT_DIR / f"results_deeponet_{grid}.json", "w") as f:
        json.dump({"best_params": best_params, "results": seed_data}, f, indent=4)
    print(f"Full results saved in {MODEL_OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--trials", type=int, default=15)
    run_study(parser.parse_args().grid, parser.parse_args().trials)
