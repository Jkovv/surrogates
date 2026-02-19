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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class DeepONet(tf.keras.Model):
    def __init__(self, filters=32, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # BRANCH 
        self.branch = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, None, 2)), # 2 frames
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters*2, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim)
        ])
        
        # TRUNK 
        self.trunk_spatial = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim)
        ])
        self.trunk_temporal = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim, activation='relu'),
            tf.keras.layers.Dense(latent_dim)
        ])

    def call(self, inputs):
        x_b, x_t = inputs
        x_b_reshaped = tf.reshape(x_b, [tf.shape(x_b)[0], tf.shape(x_b)[2], tf.shape(x_b)[3], 2])
        
        b_vec = self.branch(x_b_reshaped) 
        
        spatial_vec = self.trunk_spatial(x_t[..., :2])
        temporal_vec = self.trunk_temporal(x_t[..., 2:3])
        t_vec = spatial_vec * temporal_vec # Hadamard product (X)
        
        b_vec = tf.expand_dims(b_vec, axis=1)
        return tf.reduce_sum(b_vec * t_vec, axis=-1, keepdims=True)


def calculate_metrics_phys(y_true_phys, y_pred_phys, masks, grid_size):
    y_p = np.maximum(y_pred_phys.reshape(-1, grid_size, grid_size, 1), 0)
    y_t = y_true_phys.reshape(-1, grid_size, grid_size, 1)
    spatial_mask = np.max(masks, axis=-1, keepdims=True)
    
    # masked RMSE 
    sq_diff = np.square(y_t - y_p) * spatial_mask
    rmse = np.sqrt(np.sum(sq_diff) / (np.sum(spatial_mask) + 1e-12))
    
    # global R2
    r2 = r2_score(y_t.flatten(), y_p.flatten())
    
    dices, ssim_vals, corrs = [], [], []
    for i in range(y_t.shape[0]):
        gt, pr = y_t[i, :, :, 0], y_p[i, :, :, 0]
        
        # dice
        thresh = 0.05 * np.max(gt) if np.max(gt) > 0 else 1e-9
        yt_b, yp_b = (gt > thresh).astype(float), (pr > thresh).astype(float)
        dices.append((2.*np.sum(yt_b*yp_b)+1e-6)/(np.sum(yt_b)+np.sum(yp_b)+1e-6))
        
        # ssim
        data_range = max(gt.max() - gt.min(), 1e-9)
        ssim_vals.append(ssim(gt, pr, data_range=data_range, win_size=3))
        
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
    Y_target = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1].reshape(99, -1, 1)
    
    Y_phys_raw = np.load(data_path / "Y_raw_phys.npy").astype(np.float32)[2:, ..., idx:idx+1].reshape(99, -1, 1)
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)

    with open(data_path / "metadata.json", "r") as f:
        meta = json.load(f)
    p_min, p_max = meta["scaling"]["min"][idx], meta["scaling"]["max"][idx]

    model = DeepONet(filters=64, latent_dim=256)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-4, clipnorm=1.0), loss='mse')

    print(f"Training DeepONet: {cytokine} | Grid: {grid}")
    model.fit([X_b[:80], X_t[:80]], Y_target[:80], epochs=200, batch_size=4, verbose=1)

    Y_p_scaled = model.predict([X_b, X_t], batch_size=2)
    Y_p_phys = Y_p_scaled * (p_max - p_min) + p_min

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "results_physical": {
            "Interp": calculate_metrics_phys(Y_phys_raw[70:88], Y_p_phys[70:88], M[70:88], grid),
            "Extrap": calculate_metrics_phys(Y_phys_raw[80:99], Y_p_phys[80:99], M[80:99], grid)
        }
    }

    with open(out_dir / f"res_{suffix}.json", 'w') as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"DONE: Results saved for {suffix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.grid, args.seed, args.cytokine)
