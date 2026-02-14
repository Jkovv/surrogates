import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_deeponet_fc(grid_size, n_features, p_dim=128):
    #BRANCH NET 
    branch_in = tf.keras.layers.Input(shape=(grid_size, grid_size, n_features), name="Branch_Input")
    b = tf.keras.layers.Flatten()(branch_in)
    b = tf.keras.layers.Dense(256, activation='relu')(b)
    b = tf.keras.layers.Dense(256, activation='relu')(b)
    branch_out = tf.keras.layers.Dense(p_dim, activation='relu')(b)
    branch_out_rep = tf.keras.layers.RepeatVector(grid_size * grid_size)(branch_out)

    #TRUNK NET 
    trunk_in = tf.keras.layers.Input(shape=(grid_size * grid_size, 3), name="Trunk_Input")
    t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(trunk_in)
    t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(t)
    trunk_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(p_dim, activation='relu'))(t)

    # HADAMARD PRODUCT
    combined = tf.keras.layers.Multiply()([branch_out_rep, trunk_out])
    
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(combined)
    
    return tf.keras.Model(inputs=[branch_in, trunk_in], outputs=output)

def compute_dice_coefficient(y_true, y_pred, smooth=1e-6):
    threshold = 0.05 
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin)
    return (2. * intersection + smooth) / (union + smooth)

def calculate_metrics_phys(y_true, y_pred, masks, max_val_log):
    yt_phys = np.expm1(y_true * max_val_log)
    yp_phys = np.expm1(np.maximum(y_pred, 0) * max_val_log)
    
    spatial_mask = np.max(masks, axis=-1, keepdims=True)
    
    # global R2
    r2 = r2_score(yt_phys.flatten(), yp_phys.flatten())
    
    # masked RMSE
    sq_diff = np.square(yt_phys - yp_phys) * spatial_mask
    m_rmse = np.sqrt(np.sum(sq_diff) / (np.sum(spatial_mask) + 1e-7))

    dices, corrs, ssim_vals = [], [], []
    
    for t in range(yt_phys.shape[0]):
        gt, pr = yt_phys[t, :, :, 0], yp_phys[t, :, :, 0]
        
        # dice
        dices.append(compute_dice_coefficient(gt, pr))
        
        # SSIM
        drange = max(gt.max(), 1e-9)
        win_size = min(7, gt.shape[0], gt.shape[1])
        if win_size % 2 == 0: win_size -= 1
        ssim_vals.append(ssim(gt, pr, data_range=drange, win_size=win_size))
        
        # correlation
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            corrs.append(pearsonr(gt.flatten(), pr.flatten())[0])

    return {
        "Global_R2": float(r2),
        "Masked_RMSE": float(m_rmse),
        "Avg_Dice": float(np.mean(dices)),
        "Avg_SSIM": float(np.mean(ssim_vals)),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0
    }

def run_deeponet_experiment(grid, cytokine, seed):
    set_seed(seed)
    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir = Path("./models/deeponet")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"deeponet_fc_{cytokine.lower()}_grid{grid}_seed{seed}"
    idx = CYTOKINE_MAP[cytokine.lower()]

    X_branch = np.load(data_path / "X_branch.npy").astype(np.float32)
    X_trunk = np.load(data_path / "X_trunk.npy").astype(np.float32)
    Y_target = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M_all = np.load(data_path / "Y_masks.npy").astype(np.float32)

    with open(data_path / "scaling_params.json", "r") as f:
        scales = json.load(f)
    max_val_log = scales[cytokine.lower()]

    n = len(X_branch)
    t_end, v_end = int(0.7 * n), int(0.8 * n)

    model = build_deeponet_fc(grid, X_branch.shape[-1])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    print(f"Training DeepONet: {suffix}")
    model.fit(
        [X_branch[:t_end], X_trunk[:t_end]], 
        Y_target[:t_end].reshape(t_end, -1, 1),
        epochs=100, 
        batch_size=1 if grid >= 250 else 8,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
    )

    y_pred = model.predict([X_branch, X_trunk], batch_size=1).reshape(-1, grid, grid, 1)

    res = {
        "params": {
            "type": "DeepONet",
            "p_dim": 128,
            "lr": 0.001
        },
        "seed": seed,
        "grid": grid,
        "cytokine": cytokine.lower(),
        "results": {
            "Interpolation_72_89": calculate_metrics_phys(Y_target[70:88], y_pred[70:88], M_all[70:88], max_val_log),
            "Extrapolation_82_100": calculate_metrics_phys(Y_target[80:99], y_pred[80:99], M_all[80:99], max_val_log)
        }
    }

    with open(out_dir / f"results_{suffix}.json", 'w') as f:
        json.dump(res, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"Done. Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_deeponet_experiment(args.grid, args.cytokine, args.seed)
