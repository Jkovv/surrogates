import os, json, argparse, optuna
import numpy as np
import deepxde as dde
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from core import load_data
from validation import create_model

def evaluate_windows(model, test_data):
    X_b, coords, Y_t = test_data
    y_pred = model.predict((X_b, coords))
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    results = {}
    for name, (start, end) in windows.items():
        idx_s, idx_e = max(0, start - 80), min(len(X_b), end - 80)
        if idx_s < idx_e:
            results[name] = float(np.mean((y_pred[idx_s:idx_e] - Y_t[idx_s:idx_e])**2))
    return results

def objective(trial, grid, train, val):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 128, 256, step=64),
        'latent_dim': trial.suggest_int("latent_dim", 64, 128),
        'lr': trial.suggest_float("lr", 1e-4, 5e-4, log=True),
        'activation': trial.suggest_categorical("activation", ["tanh", "relu"])
    }
    dde.config.set_random_seed(42)
    model, _, _ = create_model(params, grid, train)
    model.compile("adam", lr=params['lr'])
    _, train_state = model.train(iterations=2000)
    loss = np.sum(train_state.best_loss_test)
    tf.keras.backend.clear_session()
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="50")
    args = parser.parse_args()

    train, val, test = load_data(args.grid)
    save_dir = f"models/pideeponet/{args.grid}x{args.grid}"
    os.makedirs(save_dir, exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args.grid, train, val), n_trials=5)

    best_p = study.best_params
    model, D_v, k_v = create_model(best_p, args.grid, train)
    model.compile("adam", lr=best_p['lr'])
    model.train(iterations=8000)

    model.save(os.path.join(save_dir, "pideeponet_model"))
    
    report = {
        "best_params": best_p,
        "learned_physics": {
            "D": [float(dde.backend.to_numpy(v)) for v in D_v],
            "k": [float(dde.backend.to_numpy(v)) for v in k_v]
        },
        "window_results": evaluate_windows(model, test)
    }
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)