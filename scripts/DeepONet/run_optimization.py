import os, sys, json, argparse, numpy as np, tensorflow as tf, optuna
from pathlib import Path
from sklearn.metrics import r2_score
from core import build_deeponet
from validation import calculate_window_metrics

try:
    from optuna.integration import TFKerasPruningCallback
except ImportError:
    TFKerasPruningCallback = None

BASE_DATA_DIR, MODEL_OUT_DIR = Path("./preprocessed"), Path("./models/deeponet")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
CYTOKINES = ["IL-8", "IL-1", "IL-6", "IL-10", "TNF", "TGF"]

def prepare_flat_data(X_b, X_t, Y, num_samples=8000):
    steps, points = X_b.shape[0], X_t.shape[1]
    actual = min(num_samples, points)
    idx = np.random.choice(points, actual, replace=False)
    return np.repeat(X_b, actual, axis=0), X_t[:, idx, :].reshape(-1, 3), Y.reshape(steps, -1, 6)[:, idx, :].reshape(-1, 6)

def run_study(grid, n_trials=15):
    path = BASE_DATA_DIR / f"{grid}x{grid}"
    X_b, X_t, Y = np.load(path/"X_branch.npy"), np.load(path/"X_trunk.npy"), np.load(path/"Y_target.npy")
    t_end, v_end = int(0.7 * len(X_b)), int(0.8 * len(X_b))

    def objective(trial):
        tf.keras.backend.clear_session()
        model = build_deeponet(trial, grid)
        # Optuna wylicza optymalny LR
        lr = trial.suggest_float("lr", 1e-5, 1.5e-3, log=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr))
        
        X_bt, X_tt, Yt = prepare_flat_data(X_b[:t_end], X_t[:t_end], Y[:t_end], 8000)
        X_bv, X_tv, Yv = prepare_flat_data(X_b[t_end:v_end], X_t[t_end:v_end], Y[t_end:v_end], 5000)
        
        callbacks = [TFKerasPruningCallback(trial, "val_loss")] if TFKerasPruningCallback else []
        model.fit([X_bt, X_tt], Yt, validation_data=([X_bv, X_tv], Yv), 
                  epochs=100, batch_size=512, verbose=0, callbacks=callbacks)
        return model.evaluate([X_bv, X_tv], Yv, verbose=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Finalny trening na najlepszych parametrach
    best = study.best_params
    print(f"Optimization complete. Best Params: {best}")
    model = build_deeponet(best, grid)
    model.compile(optimizer=tf.keras.optimizers.Adam(best["lr"]))
    X_bf, X_tf, Yf = prepare_flat_data(X_b[:t_end], X_t[:t_end], Y[:t_end], grid*grid)
    model.fit([X_bf, X_tf], Yf, epochs=200, batch_size=256, verbose=1)
    
    model.save_weights(MODEL_OUT_DIR / f"weights_deeponet_{grid}.weights.h5")

    # Raportowanie wyników dla okien czasowych
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    final_results = {"best_params": best, "per_cytokine_metrics": {}}

    for win_name, (s, e) in windows.items():
        preds = []
        for t in range(s, min(e, len(X_b))):
            p = model.predict([np.repeat(X_b[t:t+1], grid*grid, axis=0), X_t[t]], verbose=0)
            preds.append(p.reshape(grid, grid, 6))
        
        preds, true = np.array(preds), Y[s:min(e, len(X_b))]
        win_metrics = {cyto: {"R2": float(r2_score(true[..., i].flatten(), preds[..., i].flatten()))} 
                       for i, cyto in enumerate(CYTOKINES)}
        win_metrics["OVERALL"] = calculate_window_metrics(true, preds, grid)
        final_results["per_cytokine_metrics"][win_name] = win_metrics

    with open(MODEL_OUT_DIR / f"results_deeponet_{grid}.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--trials", type=int, default=15)
    run_study(parser.parse_args().grid, parser.parse_args().trials)
