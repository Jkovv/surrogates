import os, json, argparse, numpy as np, tensorflow as tf, optuna
from pathlib import Path
from core import build_deeponet
from validation import calculate_window_metrics

BASE_DATA_DIR = Path("/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/preprocessed")
MODEL_OUT_DIR = Path("/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/models/deeponet")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data(X_b, X_t, Y, Y_m, num_samples=8000):
    mask = Y_m.reshape(-1)
    active_idx = np.where(mask > 0)[0]
    n_active = int(num_samples * 0.7)
    chosen_active = np.random.choice(active_idx, min(n_active, len(active_idx)), replace=False)
    n_rem = num_samples - len(chosen_active)
    chosen_random = np.random.choice(np.arange(len(mask)), n_rem, replace=False)
    idx = np.concatenate([chosen_active, chosen_random])
    points = X_t.shape[1]
    return X_b[idx // points], X_t[idx // points, idx % points], Y.reshape(-1, 6)[idx]

def run_study(grid, n_trials=15):
    path = BASE_DATA_DIR / f"{grid}x{grid}"
    X_b, X_t, Y, Y_m = [np.load(path/f"{n}.npy") for n in ["X_branch", "X_trunk", "Y_target", "Y_masks"]]
    t_end, v_end = int(0.7 * len(X_b)), int(0.8 * len(X_b))

    def objective(trial):
        params = {
            "n_filters": trial.suggest_categorical("n_filters", [32, 64]),
            "latent_dim": trial.suggest_int("latent_dim", 128, 256),
            "trunk_width": trial.suggest_int("trunk_width", 128, 256),
            "activation": trial.suggest_categorical("activation", ["gelu", "swish"]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        }
        model = build_deeponet(params, grid)
        model.compile(optimizer=tf.keras.optimizers.Adam(params["lr"]), loss=None)
        
        X_bt, X_tt, Yt = prepare_data(X_b[:t_end], X_t[:t_end], Y[:t_end], Y_m[:t_end], 8000)
        X_bv, X_tv, Yv = prepare_data(X_b[t_end:v_end], X_t[t_end:v_end], Y[t_end:v_end], Y_m[t_end:v_end], 5000)
        
        model.fit([X_bt, X_tt], Yt, validation_data=([X_bv, X_tv], Yv), epochs=100, batch_size=512, verbose=0)
        res = model.evaluate([X_bv, X_tv], Yv, verbose=0)
        return res if not isinstance(res, list) else res[0]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best = study.best_params
    model = build_deeponet(best, grid)
    model.compile(optimizer=tf.keras.optimizers.Adam(best["lr"]), loss=None)
    
    X_bf, X_tf, Yf = prepare_data(X_b[:t_end], X_t[:t_end], Y[:t_end], Y_m[:t_end], 20000)
    model.fit([X_bf, X_tf], Yf, epochs=1000, batch_size=256, verbose=1)
    
    model.save_weights(MODEL_OUT_DIR / f"weights_deeponet_{grid}.weights.h5")
    
    windows = {"Window_82_100": (80, 99), "Window_72_89": (70, 88)}
    final_results = {"best_params": best, "per_cytokine_metrics": {}}

    for win_name, (s, e) in windows.items():
        preds = []
        for t in range(s, min(e, len(X_b))):
            p = model.predict([np.repeat(X_b[t:t+1], grid*grid, axis=0), X_t[t]], verbose=0)
            preds.append(p.reshape(grid, grid, 6))
        final_results["per_cytokine_metrics"][win_name] = calculate_window_metrics(Y[s:min(e, len(X_b))], np.array(preds), grid)

    with open(MODEL_OUT_DIR / f"results_deeponet_{grid}.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    run_study(args.grid)
