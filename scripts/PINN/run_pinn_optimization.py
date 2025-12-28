import os, json, argparse, optuna
import numpy as np
import deepxde as dde
from core_pinn import load_data_pinn
from validation_pinn import create_pinn_model

def run_window_evaluation(model, test_set):
    X_test, Y_test = test_set
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    results = {}
    
    y_pred = model.predict(X_test)
    for name, (start, end) in windows.items():
        mask = (X_test[:, 2] >= start) & (X_test[:, 2] < end)
        if mask.any():
            mse = np.mean((y_pred[mask] - Y_test[mask])**2)
            results[name] = float(mse)
    return results

def objective(trial, grid, train, val, init_phys):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 64, 512, step=64),
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'activation': trial.suggest_categorical("activation", ["tanh", "relu", "silu"])
    }
    # seed stability evaluation
    seeds, val_losses = [1, 42, 100], []
    for s in seeds:
        dde.config.set_random_seed(s)
        model, _, _ = create_pinn_model(params, grid, train, val, init_phys)
        model.compile("adam", lr=params['lr'])
        _, train_state = model.train(iterations=5000)
        val_losses.append(np.sum(train_state.best_loss))
    return np.mean(val_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="50")
    parser.add_argument("--warm_start", type=str, default=None)
    args = parser.parse_args()

    train, val, test = load_data_pinn(args.grid)
    save_dir = f"models/pinn_dde/{args.grid}x{args.grid}"
    os.makedirs(save_dir, exist_ok=True)

    init_phys = None
    if args.warm_start and os.path.exists(args.warm_start):
        with open(args.warm_start, 'r') as f:
            init_phys = json.load(f).get('learned_physics')

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args.grid, train, val, init_phys), n_trials=10)

    best_p = study.best_params
    dde.config.set_random_seed(42)
    model, D_v, k_v = create_pinn_model(best_p, args.grid, train, val, init_phys)
    model.compile("adam", lr=best_p['lr'])
    model.train(iterations=10000)

    # saving 
    model.save(os.path.join(save_dir, "pinn_model"))

    report = {
        "best_params": best_p,
        "learned_physics": {
            "D": [float(v.value) for v in D_v],
            "k": [float(v.value) for v in k_v]
        },
        "results": run_window_evaluation(model, test)
    }
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)