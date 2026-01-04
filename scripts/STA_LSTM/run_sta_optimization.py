import os, json, argparse, optuna, gc
import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from core_sta_lstm import load_data_sta, STALSTM
from validation_sta_lstm import train_and_eval_sta_lstm

def calculate_spatial_metrics(y_true, y_pred):
    thresh = np.percentile(y_true, 90)
    mask_t, mask_p = y_true > thresh, y_pred > thresh
    dice = (2. * np.logical_and(mask_t, mask_p).sum()) / (mask_t.sum() + mask_p.sum() + 1e-7)
    emd = wasserstein_distance(y_true.flatten(), y_pred.flatten())
    return float(dice), float(emd)

def evaluate_sta_windows(model, full_X, full_Y):
    preds = model.predict(full_X, batch_size=1, verbose=0)
    
    windows = {"Window_82_100": (82, 101), "Window_72_89": (72, 90)}
    report = {}
    
    for name, (start, end) in windows.items():
        idx_s, idx_e = int(start), min(len(full_Y), int(end))
        
        w_preds, w_targets = preds[idx_s:idx_e], full_Y[idx_s:idx_e]
        dice_l, emd_l = [], []
        
        for i in range(len(w_targets)):
            d, e = calculate_spatial_metrics(w_targets[i], w_preds[i])
            dice_l.append(d); emd_l.append(e)
            
        report[name] = {
            "RMSE": float(np.sqrt(np.mean((w_preds - w_targets)**2))),
            "Dice": float(np.mean(dice_l)),
            "EMD": float(np.mean(emd_l)),
            "R2_Trajectory": float(r2_score(np.mean(w_targets, axis=(1, 2, 3)), 
                                            np.mean(w_preds, axis=(1, 2, 3))))
        }
    return report

def objective(trial, train_set, val_set, grid_size):
    params = {
        'hidden_size': trial.suggest_int("hidden_size", 32, 128, step=32),
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'activation': trial.suggest_categorical("activation", ["ReLU", "SiLU", "Tanh"])
    }
    model = train_and_eval_sta_lstm(params, train_set, val_set, 42, grid_size)
    preds = model.predict(val_set[0], batch_size=1, verbose=0)
    loss = np.mean((val_set[1] - preds)**2)
    tf.keras.backend.clear_session(); gc.collect()
    return float(loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--grid", type=int, default=50)
    args = parser.parse_args()
    
    save_dir = f"models/sta_lstm/{args.grid}x{args.grid}"; os.makedirs(save_dir, exist_ok=True)
    train, val, test = load_data_sta(args.grid)
    
    path = f"preprocessed/{args.grid}x{args.grid}"
    full_X = np.load(os.path.join(path, "X_lstm.npy")).astype(np.float32)
    full_Y = np.load(os.path.join(path, "Y_target.npy")).astype(np.float32)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, train, val, args.grid), n_trials=5)

    seeds = [1, 42, 100]
    all_results = []
    
    for s in seeds:
        model = train_and_eval_sta_lstm(study.best_params, train, val, s, args.grid)
        
        model.save_weights(os.path.join(save_dir, f"model_seed_{s}.h5"))
        
        train_mse = float(np.mean((model.predict(train[0], verbose=0) - train[1])**2))
        val_mse = float(np.mean((model.predict(val[0], verbose=0) - val[1])**2))
        
        window_data = evaluate_sta_windows(model, full_X, full_Y)
        
        all_results.append({
            "seed": s,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "windows": window_data
        })
        tf.keras.backend.clear_session(); gc.collect()

    report = {
        "model": "sta-lstm",
        "grid": args.grid,
        "best_params": study.best_params,
        "detailed_seeds": all_results
    }
    with open(os.path.join(save_dir, "research_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"STA-LSTM finished for grid {args.grid}.")
