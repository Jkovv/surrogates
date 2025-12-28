import os, json, argparse, optuna
import numpy as np
from core import load_data_deeponet
from validation import train_and_eval

def evaluate_windows(model, test_set, coords):
    X_b_test, Y_t_test = test_set
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    results = {}
    
    for name, (start, end) in windows.items():
        xb_win = X_b_test[:, start:end]
        yt_win = Y_t_test[:, start:end]
        
        num_sim, num_t = xb_win.shape[0], xb_win.shape[1]
        xb_flat = np.repeat(xb_win.reshape(-1, xb_win.shape[-1]), coords.shape[0], axis=0)
        xt_flat = np.tile(coords, (num_sim * num_t, 1))
        
        pred = model.predict((xb_flat, xt_flat))
        mse = np.mean((pred.reshape(yt_win.shape) - yt_win)**2)
        results[name] = float(mse)
    return results

def objective(trial, train, val, b_dim, t_dim):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 128, 512),
        'latent_dim': trial.suggest_int("latent_dim", 64, 256), 
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'activation': trial.suggest_categorical("activation", ["tanh", "relu", "silu"]),
        'epochs': 10000 
    }
    # avg loss across diff seeds: 1,42,100
    seeds = [1, 42, 100]
    losses = [train_and_eval(params, train, val, b_dim, t_dim, s)[0] for s in seeds]
    return np.mean(losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="50")
    args = parser.parse_args()

    train, val, test, coords = load_data_deeponet(int(args.grid))
    b_dim, t_dim = train[0][0].shape[1], coords.shape[1]

    # hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, val, b_dim, t_dim), n_trials=10)

    # training with best params
    best_p = study.best_params
    best_p['epochs'] = 20000 
    _, final_model = train_and_eval(best_p, train, val, b_dim, t_dim, seed=42)

    # saving results
    save_path = f"models/deeponet_dde/{args.grid}x{args.grid}"
    os.makedirs(save_path, exist_ok=True)
    report = {"best_params": best_p, "window_results": evaluate_windows(final_model, test, coords)}
    
    with open(os.path.join(save_path, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print(f"DeepONetoptimization for {args.grid}x{args.grid} complete.")