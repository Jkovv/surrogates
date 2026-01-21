import os, json, argparse, numpy as np, tensorflow as tf, optuna
from pathlib import Path
from core import STALSTMUncertainty, set_all_seeds
from validation import calculate_window_metrics

BASE_DATA_DIR = Path("/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/preprocessed")
OUT_DIR = Path("/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/models/sta_lstm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_study(grid, n_trials=15, seed=42):
    set_all_seeds(seed)
    
    path = BASE_DATA_DIR / f"{grid}x{grid}"
    X = np.load(path / "X_lstm.npy")
    Y = np.load(path / "Y_target.npy")
    
    # 70/10/20
    t_end, v_end = int(0.7 * len(X)), int(0.8 * len(X))

    def objective(trial):
        params = {
            "n_filters": trial.suggest_categorical("n_filters", [32, 64]),
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512),
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        }
        
        model = STALSTMUncertainty(grid, n_filters=params["n_filters"], hidden_dim=params["hidden_dim"])
        model.compile(optimizer=tf.keras.optimizers.Adam(params["lr"]), loss=None)
        
        model.fit(X[:t_end], Y[:t_end], 
                  validation_data=(X[t_end:v_end], Y[t_end:v_end]), 
                  epochs=50, batch_size=32, verbose=0)
        
        return model.evaluate(X[t_end:v_end], Y[t_end:v_end], verbose=0)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    
    best = study.best_params
    model = STALSTMUncertainty(grid, n_filters=best["n_filters"], hidden_dim=best["hidden_dim"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best["lr"]), loss=None)
    
    print(f"Final training for seed {seed}...")
    model.fit(X[:v_end], Y[:v_end], epochs=500, batch_size=16, verbose=1)
    
    model.save_weights(OUT_DIR / f"weights_stalstm_{grid}_seed{seed}.weights.h5")

    windows = {"Window_82_100": (80, 100), "Window_72_89": (70, 89)}
    results = {"best_params": best, "seed": seed, "per_cytokine_metrics": {}}
    
    for win, (s, e) in windows.items():
        preds = model.predict(X[s:e])
        results["per_cytokine_metrics"][win] = calculate_window_metrics(Y[s:e], preds, grid)

    with open(OUT_DIR / f"results_stalstm_{grid}_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_study(args.grid, seed=args.seed)
