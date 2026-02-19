import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
import optuna
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

class DeepONet(tf.keras.Model):
    def __init__(self, filters=32, latent_dim=128):
        super().__init__()
        # BRANCH
        self.branch = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, None, 2)),
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters*2, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim)
        ])
        
        # TRUNK
        self.trunk = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim, activation='tanh'),
            tf.keras.layers.Dense(latent_dim, activation='tanh'),
            tf.keras.layers.Dense(latent_dim, activation='tanh'),
            tf.keras.layers.Dense(latent_dim)
        ])

    def call(self, inputs):
        x_b, x_t = inputs
        batch_size = tf.shape(x_b)[0]
        h = tf.shape(x_b)[2]
        w = tf.shape(x_b)[3]
        
        x_b_reshaped = tf.reshape(x_b, [batch_size, h, w, 2])
        
        b_vec = self.branch(x_b_reshaped) 
        t_vec = self.trunk(x_t)           
        
        b_vec = tf.expand_dims(b_vec, axis=1)
        return tf.reduce_sum(b_vec * t_vec, axis=-1, keepdims=True)

def calculate_metrics_phys(y_true, y_pred, masks, grid_size):
    y_true_img = y_true.reshape(-1, grid_size, grid_size, 1)
    y_pred_img = y_pred.reshape(-1, grid_size, grid_size, 1)
    spatial_mask = np.max(masks, axis=-1, keepdims=True)
    y_pred_img = np.maximum(y_pred_img, 0)
    
    # msked RMSE 
    sq_diff = np.square(y_true_img - y_pred_img) * spatial_mask
    rmse = np.sqrt(np.sum(sq_diff) / (np.sum(spatial_mask) + 1e-12))
    
    # global R2
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    dices, ssim_vals, corrs = [], [], []
    for t in range(y_true_img.shape[0]):
        gt, pr = y_true_img[t,:,:,0], y_pred_img[t,:,:,0]
        # Dice
        thresh = 0.05 * np.max(gt) if np.max(gt) > 0 else 1e-9
        yt_b, yp_b = (gt > thresh).astype(float), (pr > thresh).astype(float)
        dices.append((2.*np.sum(yt_b*yp_b)+1e-6)/(np.sum(yt_b)+np.sum(yp_b)+1e-6))
        # SSIM
        dr = max(gt.max() - gt.min(), 1e-9)
        ssim_vals.append(ssim(gt, pr, data_range=dr, win_size=3))
        # Spatial Correlation
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            corrs.append(pearsonr(gt.flatten(), pr.flatten())[0])
            
    return {
        "Global_R2": float(r2), 
        "Masked_RMSE_Phys": float(rmse),
        "Avg_Dice": float(np.mean(dices)), 
        "Avg_SSIM": float(np.mean(ssim_vals)),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0
    }

def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir = Path(f"./models/deeponet")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    idx = CYTOKINE_MAP[cytokine.lower()]
    X_b = np.load(data_path / "X_lstm.npy").astype(np.float32)[..., idx:idx+1]
    X_t = np.load(data_path / "X_trunk.npy").astype(np.float32)
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    Y_flat = Y.reshape(Y.shape[0], -1, 1)
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)

    with open(data_path / "metadata.json", "r") as f:
        meta = json.load(f)
    p_min, p_max = meta["scaling"]["min"][idx], meta["scaling"]["max"][idx]

    def weighted_loss(yt, yp):
        w = tf.cast(yt > 0.01, tf.float32) * 50.0 + 1.0
        return tf.reduce_mean(w * tf.square(yt - yp))

    def objective(trial):
        tf.keras.backend.clear_session()
        f = trial.suggest_categorical("filters", [16, 32])
        ld = trial.suggest_categorical("latent", [64, 128])
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        m = DeepONet(filters=f, latent_dim=ld)
        m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=weighted_loss)
        m.fit([X_b[:70], X_t[:70]], Y_flat[:70], epochs=10, batch_size=4, verbose=0)
        return m.evaluate([X_b[70:80], X_t[70:80]], Y_flat[70:80], verbose=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best = study.best_params
    
    final_model = DeepONet(filters=best["filters"], latent_dim=best["latent"])
    final_model.compile(optimizer=tf.keras.optimizers.Adam(best["lr"], clipnorm=1.0), loss=weighted_loss)
    final_model.fit([X_b[:80], X_t[:80]], Y_flat[:80], epochs=150, batch_size=4, verbose=1)
    
    Y_pred_flat = final_model.predict([X_b, X_t])
    Y_p_phys = Y_pred_flat * (p_max - p_min) + p_min
    Y_a_phys = Y_flat * (p_max - p_min) + p_min
    
    suffix = f"{cytokine}_{grid}_{seed}"
    res = {
        "params": best, "grid": grid, "seed": seed, "cytokine": cytokine,
        "results_phys": {
            "Interpolation_72_89": calculate_metrics_phys(Y_a_phys[70:88], Y_p_phys[70:88], M[70:88], grid),
            "Extrapolation_82_100": calculate_metrics_phys(Y_a_phys[80:99], Y_p_phys[80:99], M[80:99], grid)
        }
    }
    
    with open(out_dir / f"res_{suffix}.json", 'w') as f:
        json.dump(res, f, indent=4)
    final_model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"DONE: {suffix} results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.grid, args.seed, args.cytokine)
