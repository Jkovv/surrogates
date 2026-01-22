import os, json, argparse, numpy as np, tensorflow as tf, optuna
import random
from pathlib import Path
from core import build_deeponet
from validation import calculate_window_metrics

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

BASE_DATA_DIR = Path("/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/preprocessed")
MODEL_OUT_DIR = Path("/gpfs/scratch1/shared/jkowalczuk/surrogates/burns/models/deeponet")
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data(X_b, X_t, Y, Y_m, num_samples=10000, seed=42):
    np.random.seed(seed)
    mask = Y_m.reshape(-1)
    active_idx = np.where(mask > 0)[0]
    n_active = int(num_samples * 0.8)
    chosen_active = np.random.choice(active_idx, min(n_active, len(active_idx)), replace=False)
    n_rem = num_samples - len(chosen_active)
    chosen_random = np.random.choice(np.arange(len(mask)), n_rem, replace=False)
    idx = np.concatenate([chosen_active, chosen_random])
    points = X_t.shape[1]
    return X_b[idx // points], X_t[idx // points, idx % points], Y.reshape(-1, 6)[idx]

def run_study(grid, n_trials=15, seed=42):
    path = BASE_DATA_DIR / f"{grid}x{grid}"
    X_b, X_t, Y, Y_m = [np.load(path/f"{n}.npy") for n in ["X_branch", "X_trunk", "Y_target", "Y_masks"]]
    t_end, v_end = int(0.7 * len(X_b)), int(0.8 * len(X_b))

    def objective(trial):
        params = {
            "n_filters": trial.suggest_categorical("n_filters", [32, 64]),
            "latent_dim": trial.suggest_int("latent_dim", 128, 256),
            "trunk_width": trial.suggest_int("trunk_width", 128, 256),
            "activation": trial.suggest_categorical("activation", ["swish"]), 
            "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        }
        model = build_deeponet(params, grid, seed=seed)
        model.compile(optimizer=tf.keras.optimizers.Adam(params["lr"]), loss=None)
        
        X_bt, X_tt, Yt = prepare_data(X_b[:t_end], X_t[:t_end], Y[:t_end], Y_m[:t_end], 12000, seed=seed)
        X_bv, X_tv, Yv = prepare_data(X_b[t_end:v_end], X_t[t_end:v_end], Y[t_end:v_end], Y_m[t_end:v_end], 6000, seed=seed)
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit([X_bt, X_tt], Yt, validation_data=([X_bv, X_tv], Yv), 
                  epochs=150, batch_size=512, verbose=0, callbacks=[es])
        
        val_loss = model.evaluate([X_bv, X_tv], Yv, verbose=0)
        return val_loss if not np.isnan(val_loss) else 1e9

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    
    best = study.best_params
    model = build_deeponet(best, grid, seed=seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(best["lr"]), loss=None)
    
    X_bf, X_tf, Yf = prepare_data(X_b[:v_end], X_t[:v_end], Y[:v_end], Y_m[:v_end], 40000, seed=seed)
    model.fit([X_bf, X_tf], Yf, epochs=800, batch_size=256, verbose=1)
    
    model.save_weights(MODEL_OUT_DIR / f"weights_stable_deeponet_{grid}_seed{seed}.weights.h5")
    
    windows = {"Window_82_100": (80, 100), "Window_72_89": (70, 89)}
    results = {"best_params": best, "per_cytokine_metrics": {}}

    for win_name, (s, e) in windows.items():
        preds = []
        for t in range(s, min(e, len(X_b))):
            p = model.predict([np.repeat(X_b[t:t+1], grid*grid, axis=0), X_t[t]], verbose=0)
            preds.append(p.reshape(grid, grid, 6))
        results["per_cytokine_metrics"][win_name] = calculate_window_metrics(Y[s:min(e, len(X_b))], np.array(preds), grid)

    with open(MODEL_OUT_DIR / f"results_stable_deeponet_{grid}_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_all_seeds(args.seed) 
    run_study(args.grid, seed=args.seed)
