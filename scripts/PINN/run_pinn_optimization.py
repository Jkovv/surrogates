import os, json, argparse, optuna
import numpy as np
import deepxde as dde
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from core_pinn import load_data_pinn
from validation_pinn import create_pinn_model

def run_window_evaluation(model, test_set):
    X_t, Y_t = test_set
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    y_p = model.predict(X_t)
    results = {}
    for n, (s, e) in windows.items():
        mask = (X_t[:,2] >= s) & (X_t[:,2] < e)
        if np.any(mask):
            results[n] = float(np.mean((y_p[mask] - Y_t[mask])**2))
    return results

def objective(trial, grid, train, val, init_phys):
    params = {'hidden_size': trial.suggest_int("hidden_size", 128, 256, step=64),
              'lr': trial.suggest_float("lr", 1e-4, 5e-4, log=True),
              'activation': trial.suggest_categorical("activation", ["tanh", "relu"])}
    dde.config.set_random_seed(42)
    model, _, _ = create_pinn_model(params, grid, train, val, init_phys)
    model.compile("adam", lr=params['lr'])
    _, train_state = model.train(iterations=2000, batch_size=256) 
    loss = np.sum(train_state.best_loss_test)
    tf.keras.backend.clear_session()
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="50")
    args = parser.parse_args()

    train, val, test = load_data_pinn(args.grid)
    save_dir = f"models/pinn_dde/{args.grid}x{args.grid}"
    os.makedirs(save_dir, exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args.grid, train, val, None), n_trials=5)

    best_p = study.best_params
    model, D_v, k_v = create_pinn_model(best_p, args.grid, train, val, None)
    model.compile("adam", lr=best_p['lr'])
    model.train(iterations=8000, batch_size=256) 

    model.save(os.path.join(save_dir, "pinn_model"))
    
    report = {
        "best_params": best_p,
        "learned_physics": {
            "D": [float(dde.backend.to_numpy(v)) for v in D_v],
            "k": [float(dde.backend.to_numpy(v)) for v in k_v]
        },
        "results": run_window_evaluation(model, test)
    }
    
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"PINN optimization for grid {args.grid} finished successfully.")