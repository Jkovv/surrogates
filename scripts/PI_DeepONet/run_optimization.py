import os, json, argparse, optuna, gc, sys
import numpy as np
import tensorflow as tf
import deepxde as dde
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core import load_data_pideeponet
from validation import create_pideeponet_model

def calculate_spatial_metrics(y_true, y_pred):
    thresh = np.percentile(y_true, 90)
    mask_t, mask_p = y_true > thresh, y_pred > thresh
    dice = (2. * np.logical_and(mask_t, mask_p).sum()) / (mask_t.sum() + mask_p.sum() + 1e-7)
    emd = wasserstein_distance(y_true.flatten(), y_pred.flatten())
    return dice, emd

def evaluate_windows(model, test_tuple, coords):
    X_b_test, Y_t_test = test_tuple
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    report = {}
    for name, (start, end) in windows.items():
        idx_s, idx_e = max(0, start - 82), min(len(X_b_test), end - 82)
        if idx_s >= idx_e: continue
        preds = model.predict((X_b_test[idx_s:idx_e], coords))
        targets = Y_t_test[idx_s:idx_e]
        d_l = [calculate_spatial_metrics(targets[i], preds[i])[0] for i in range(len(targets))]
        e_l = [calculate_spatial_metrics(targets[i], preds[i])[1] for i in range(len(targets))]
        r2_traj = r2_score(np.mean(targets, axis=(1, 2)), np.mean(preds, axis=(1, 2)))
        report[name] = {"RMSE": float(np.sqrt(np.mean((preds - targets)**2))),
                        "Dice": float(np.mean(d_l)), "EMD": float(np.mean(e_l)),
                        "R2_Trajectory": float(r2_traj)}
    return report

def objective(trial, grid, train, val, coords):
    params = {'hidden_size': trial.suggest_int("hidden_size", 128, 256, step=64),
              'latent_dim': trial.suggest_int("latent_dim", 64, 128),
              'lr': trial.suggest_float("lr", 1e-4, 5e-4, log=True),
              'activation': trial.suggest_categorical("activation", ["tanh", "relu"])}
    tf.keras.backend.clear_session(); gc.collect()
    model = create_pideeponet_model(params, grid, train, val, coords)
    model.compile("adam", lr=params['lr'])
    _, train_state = model.train(iterations=1000)
    loss = np.sum(train_state.best_loss_test)
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    train, val, test, coords = load_data_pideeponet(args.grid)
    save_dir = f"models/pi_deeponet_dde/{args.grid}x{args.grid}"; os.makedirs(save_dir, exist_ok=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args.grid, train, val, coords), n_trials=5)
    seeds, best_p = [1, 42, 100], study.best_params
    all_results = []
    for s in seeds:
        tf.keras.backend.clear_session(); gc.collect()
        dde.config.set_random_seed(s); tf.keras.utils.set_random_seed(s)
        model = create_pideeponet_model(best_p, args.grid, train, val, coords)
        model.compile("adam", lr=best_p['lr'])
        model.train(iterations=5000, display_every=1000)
        model.save(os.path.join(save_dir, f"model_seed_{s}"))
        all_results.append({"seed": s, "windows": evaluate_windows(model, test, coords)})
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump({"model": "pi_deeponet", "grid": args.grid, "detailed_seeds": all_results}, f, indent=4)