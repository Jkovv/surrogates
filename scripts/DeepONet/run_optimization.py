import os
import json
import torch
import torch.nn as nn
import numpy as np
import optuna
import argparse
from core import load_data_methodology, DeepONet, DEVICE
from validation import train_and_eval

def run_window_evaluation(model, test_set, coords):
    model.eval()
    X_test, Y_test = test_set
    criterion = nn.MSELoss()
    
    windows = {
        "Window_82_100": (82, 101), #(start, end+1) 
        "Window_72_89": (72, 90)
    }
    
    results = {}
    with torch.no_grad():
        for name, (start, end) in windows.items():
            x_win = torch.tensor(X_test[:, start:end]).to(DEVICE)
            y_win = torch.tensor(Y_test[:, start:end]).to(DEVICE)
            pred = model(x_win, torch.tensor(coords).to(DEVICE))
            results[name] = criterion(pred, y_win).item()
            
    return results

def objective(trial, train_set, val_set, coords, grid_size):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 128, 512),
        'latent_dim': trial.suggest_int("latent_dim", 60, 240, step=60), 
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'act_branch': trial.suggest_categorical("act_branch", ["ReLU", "GELU", "SiLU"]),
        'act_trunk': trial.suggest_categorical("act_trunk", ["Tanh", "GELU", "SiLU"])
    }

    # seeds 1, 42, 100
    seeds = [1, 42, 100]
    seed_losses = [train_and_eval(params, train_set, val_set, coords, s)[0] for s in seeds]
    return np.mean(seed_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=str, default="50")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    grid_configs = {"50": 30, "100": 20, "250": 15, "500": 10} # maybe change?
    tasks = grid_configs.keys() if args.grid == "all" else [args.grid]

    for current_grid in tasks:
        grid_val = int(current_grid)
        n_trials = args.trials or grid_configs[current_grid]
        save_dir = f"models/deeponet/{grid_val}x{grid_val}"
        os.makedirs(save_dir, exist_ok=True)

        train, val, test, coords = load_data_methodology(grid_val) # 70/10/20

        # optuna - hyperparams
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: objective(t, train, val, coords, grid_val), n_trials=n_trials)

        # final model 
        best_p = study.best_params
        _, final_state = train_and_eval(best_p, train, val, coords, seed=42)
        
        final_model = DeepONet(train[0].shape[2], coords.shape[1], 
                              best_p['latent_dim'], best_p['hidden_size'], 
                              best_p['act_branch'], best_p['act_trunk']).to(DEVICE)
        final_model.load_state_dict(final_state)

        # raport
        win_report = run_window_evaluation(final_model, test, coords)
        
        with open(os.path.join(save_dir, "research_report.json"), "w") as f:
            json.dump({"best_params": best_p, "results": win_report}, f, indent=4)
        
        torch.save(final_state, os.path.join(save_dir, "final_model.pth"))
        print(f"Done: {grid_val}x{grid_val}")