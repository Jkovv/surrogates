import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
import deepxde as dde
import optuna
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore")
tf.keras.backend.set_floatx('float32')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def compute_dice_coefficient(y_true, y_pred, smooth=1e-6):
    threshold = 0.05 
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin)
    return (2. * intersection + smooth) / (union + smooth)

def calculate_metrics(y_true, y_pred, masks):
    if y_true.ndim == 4: y_true = y_true[..., 0]
    if y_pred.ndim == 4: y_pred = y_pred[..., 0]
    
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    spatial_mask = np.max(masks, axis=-1).squeeze()
    sq_diff = np.square(y_true - y_pred) * spatial_mask
    m_rmse = np.sqrt(np.sum(sq_diff) / (np.sum(spatial_mask) + 1e-7))
    
    dices, corrs, ssim_vals = [], [], []
    for t in range(y_true.shape[0]):
        gt, pr = y_true[t], y_pred[t]
        dices.append(compute_dice_coefficient(gt, pr))
        drange = max(gt.max(), 1.0)
        win_size = min(7, gt.shape[0], gt.shape[1])
        if win_size % 2 == 0: win_size -= 1
        ssim_vals.append(ssim(gt, pr, data_range=drange, win_size=win_size))
        if np.std(gt) > 1e-9 and np.std(pr) > 1e-9:
            corrs.append(pearsonr(gt.flatten(), pr.flatten())[0])
            
    return {
        "Global_R2": float(r2), 
        "Masked_RMSE": float(m_rmse),
        "Avg_Dice": float(np.mean(dices)),
        "Avg_SSIM": float(np.mean(ssim_vals)),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0
    }

def create_deeponet_data(grid, cytokine):
    base_dir = Path(".")
    data_path = base_dir / f"preprocessed/{grid}x{grid}"
    Y_all_scaled = np.load(data_path / "Y_target.npy").astype(np.float32)
    M_all = np.load(data_path / "Y_masks.npy").astype(np.float32)
    idx = CYTOKINE_MAP[cytokine]
    Y_target_scaled = Y_all_scaled[..., idx:idx+1]
    T_MAX = Y_target_scaled.shape[0] - 1
    u_0 = Y_target_scaled[0].flatten()
    dim_branch = len(u_0)
    
    x_coords = np.linspace(0, 1, grid)
    y_coords = np.linspace(0, 1, grid)
    t_coords = np.linspace(0, 1, T_MAX + 1)
    X, Y, T = np.meshgrid(x_coords, y_coords, t_coords, indexing='ij')
    X_trunk = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=-1)
    
    N_points = X_trunk.shape[0]
    X_branch_tiled = np.tile(u_0, (N_points, 1))
    X_train = np.hstack([X_branch_tiled, X_trunk])
    Y_train_scaled = np.expand_dims(Y_target_scaled.flatten(), axis=-1)
    
    return X_train, Y_train_scaled, dim_branch, Y_target_scaled, M_all, T_MAX, data_path

class SingleInputDeepONet(dde.maps.DeepONet):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer, split_idx):
        super().__init__(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer)
        self.split_idx = split_idx
    def call(self, inputs, training=False):
        x_branch = inputs[:, :self.split_idx]
        x_trunk = inputs[:, self.split_idx:]
        return super().call((x_branch, x_trunk), training=training)

def run_deeponet(grid, seed, cytokine):
    set_seed(seed)
    base_dir = Path(".")
    out_dir = base_dir / "models/deeponet"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_full, Y_train_full, dim_branch, Y_target_scaled, M_all, T_MAX, data_path = create_deeponet_data(grid, cytokine)

    sub_idx = np.random.choice(len(X_train_full), int(0.1 * len(X_train_full)), replace=False)
    data_sub = dde.data.DataSet(X_train=X_train_full[sub_idx], y_train=Y_train_full[sub_idx], 
                                X_test=X_train_full[sub_idx], y_test=Y_train_full[sub_idx])

    def objective(trial):
        tf.keras.backend.clear_session()
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        width = trial.suggest_categorical("neurons", [64, 128])
        depth = trial.suggest_int("layers", 2, 4)
        
        net = SingleInputDeepONet([dim_branch] + [width] * depth + [64], [3] + [width] * depth + [64], 
                                  "relu", "Glorot uniform", split_idx=dim_branch)
        net.apply_output_transform(lambda x, y: tf.nn.softplus(y))
        model = dde.Model(data_sub, net)
        model.compile("adam", lr=lr)
        model.train(iterations=1000, batch_size=2048, display_every=1000)
        return model.train_state.loss_train[0]

    print("optuna (for subsampled 10%):")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best = study.best_params
    print(f"Best Params: {best}")

    tf.keras.backend.clear_session()
    
    width, depth = best['neurons'], best['layers']
    suffix = f"{cytokine}_grid{grid}_seed{seed}_arch{width}x{depth}"
    
    net = SingleInputDeepONet([dim_branch] + [width] * depth + [64], 
                              [3] + [width] * depth + [64], 
                              "relu", "Glorot uniform", split_idx=dim_branch)
    net.apply_output_transform(lambda x, y: tf.nn.softplus(y))
    
    data_full = dde.data.DataSet(X_train=X_train_full, y_train=Y_train_full, X_test=X_train_full, y_test=Y_train_full)
    model = dde.Model(data_full, net)
    model.compile("adam", lr=best['lr'])
    
    ckpt_cb = dde.callbacks.ModelCheckpoint(str(out_dir / f"model_{suffix}.ckpt"), save_better_only=True, period=1000)
    
    print(f"full run:")
    model.train(iterations=15000, batch_size=10240, display_every=500, callbacks=[ckpt_cb])
    
    with open(data_path / "scaling_params.json", "r") as f:
        scaling_params = json.load(f)
    max_val_log = scaling_params[cytokine]

    y_pred_scaled = model.predict(X_train_full).reshape(T_MAX + 1, grid, grid, 1)
    
    # exp(y * max) - 1
    y_pred_phys = np.expm1(y_pred_scaled * max_val_log)
    y_true_phys = np.expm1(Y_target_scaled * max_val_log)

    res = {
        "params": best, 
        "seed": seed,
        "grid": grid,
        "cytokine": cytokine,
        "metrics": {
            "Interpolation_72_89": calculate_metrics(y_true_phys[72:89], y_pred_phys[72:89], M_all[72:89]),
            "Extrapolation_82_100": calculate_metrics(y_true_phys[82:100], y_pred_phys[82:100], M_all[82:100])
        }
    }
    
    with open(out_dir / f"results_{suffix}.json", 'w') as f:
        json.dump(res, f, indent=4)
        
    net.save_weights(str(out_dir / f"weights_{suffix}.weights.h5"))
    print(f"Saved results_{suffix}.json with physical metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_deeponet(args.grid, args.seed, args.cytokine.lower())
