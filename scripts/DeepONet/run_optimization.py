import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json, argparse, numpy as np, tensorflow as tf, optuna
from pathlib import Path

try:
    from optuna_integration import TFKerasPruningCallback
except ImportError:
    try:
        from optuna.integration import TFKerasPruningCallback
    except ImportError:
        TFKerasPruningCallback = None

from core import build_deeponet
from validation import calculate_window_metrics

BASE_DATA_DIR, MODEL_OUT_DIR = Path("./preprocessed"), Path("./models/deeponet")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
CYTOKINES = ["IL-8", "IL-1", "IL-6", "IL-10", "TNF", "TGF"]

def balanced_weighted_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    spatial_weight = tf.where(tf.greater(y_true, 1e-5), 10.0, 1.0)
    channel_losses = tf.reduce_mean(error * spatial_weight, axis=[0]) 
    return tf.reduce_mean(tf.math.log(channel_losses + 1e-7))

def prepare_flat_data(X_b, X_t, Y, num_samples=5000):
    steps, points = X_b.shape[0], X_t.shape[1]
    actual = min(num_samples, points)
    idx = np.random.choice(points, actual, replace=False)
    return np.repeat(X_b, actual, axis=0), X_t[:, idx, :].reshape(-1, 3), Y.reshape(steps, -1, 6)[:, idx, :].reshape(-1, 6)

def run_study(grid, n_trials=15):
    print(f"\nStarting optimization: {grid}x{grid}")
    path = BASE_DATA_DIR / f"{grid}x{grid}"
    X_b, X_t, Y = np.load(path/"X_branch.npy"), np.load(path/"X_trunk.npy"), np.load(path/"Y_target.npy")
    t_end, v_end = int(0.7 * len(X_b)), int(0.8 * len(X_b))

    def objective(trial):
        tf.keras.backend.clear_session()
        model = build_deeponet(trial, grid)
        lr = trial.suggest_float("lr", 1e-5, 1.5e-4, log=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=balanced_weighted_mse)
        X_bt, X_tt, Yt = prepare_flat_data(X_b[:t_end], X_t[:t_end], Y[:t_end], 5000)
        X_bv, X_tv, Yv = prepare_flat_data(X_b[t_end:v_end], X_t[t_end:v_end], Y[t_end:v_end], 5000)
        
        callbacks = [TFKerasPruningCallback(trial, "val_loss")] if TFKerasPruningCallback else []
        model.fit([X_bt, X_tt], Yt, validation_data=([X_bv, X_tv], Yv), epochs=100, batch_size=256, verbose=0,
                  callbacks=callbacks)
        return model.evaluate([X_bv, X_tv], Yv, verbose=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"\nBest params: {best}")
    model = build_deeponet(best, grid)
    model.compile(optimizer=tf.keras.optimizers.Adam(best["lr"]), loss=balanced_weighted_mse)
    X_bf, X_tf, Yf = prepare_flat_data(X_b[:t_end], X_t[:t_end], Y[:t_end], grid*grid)
    model.fit([X_bf, X_tf], Yf, epochs=150, batch_size=128, verbose=1)
    
    model.save_weights(MODEL_OUT_DIR / f"weights_deeponet_{grid}_balanced.weights.h5")

    windows = {"Window_82_100": (82, 100), "Window_72_89": (72, 89)}
    final_results = {"best_params": best, "per_cytokine_metrics": {}}

    for win_name, (s, e) in windows.items():
        idx_s, idx_e = min(s, len(X_b)-1), min(e, len(X_b))
        preds = []
        for t in range(idx_s, idx_e):
            p = model.predict([np.repeat(X_b[t:t+1], grid*grid, axis=0), X_t[t]], verbose=0)
            preds.append(p.reshape(grid, grid, 6))
        preds, true = np.array(preds), Y[idx_s:idx_e]

        win_metrics = {}
        for i, cyto in enumerate(CYTOKINES):
            r2 = r2_score(true[..., i].flatten(), preds[..., i].flatten())
            win_metrics[cyto] = {"R2": float(r2)}
        
        win_metrics["OVERALL"] = calculate_window_metrics(true, preds, grid)
        final_results["per_cytokine_metrics"][win_name] = win_metrics

    with open(MODEL_OUT_DIR / f"results_deeponet_{grid}.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Results saved in {MODEL_OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--trials", type=int, default=15)
    run_study(parser.parse_args().grid, parser.parse_args().trials)
