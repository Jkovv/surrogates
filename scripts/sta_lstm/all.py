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

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=64):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = 16
        
        # Encoder 
        self.enc = tf.keras.layers.TimeDistributed(tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D()
        ]))
        
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        
        # Decoder
        self.dec = tf.keras.Sequential([
            tf.keras.layers.Dense(self.latent_dim * self.latent_dim * filters, activation='relu'),
            tf.keras.layers.Reshape((self.latent_dim, self.latent_dim, filters)),
            tf.keras.layers.Conv2DTranspose(filters//2, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters//4, 3, strides=2, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='linear'),
            tf.keras.layers.Resizing(grid_size, grid_size)
        ])

    def call(self, x):
        return self.dec(self.lstm(self.enc(x)))

def calculate_metrics_phys(y_true, y_pred, masks, grid_size):
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    y_t = y_true[:min_t]
    y_p = y_pred[:min_t]
    m_s = np.max(masks[:min_t], axis=-1, keepdims=True)
    
    y_p = np.maximum(y_p, 0)
    
    # Masked RMSE
    sq_diff = np.square(y_t - y_p) * m_s
    rmse = np.sqrt(np.sum(sq_diff) / (np.sum(m_s) + 1e-12))
    r2 = r2_score(y_t.flatten(), y_p.flatten())
    
    dices, corrs = [], []
    for t in range(min_t):
        gt, pr = y_t[t,:,:,0], y_p[t,:,:,0]
        thresh = 0.05 * np.max(gt) if np.max(gt) > 0 else 1e-9
        # Dice
        yt_b, yp_b = (gt > thresh).astype(float), (pr > thresh).astype(float)
        dices.append((2.*np.sum(yt_b*yp_b)+1e-6)/(np.sum(yt_b)+np.sum(yp_b)+1e-6))
        # Correlation
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            corrs.append(pearsonr(gt.flatten(), pr.flatten())[0])
            
    return {
        "Global_R2": float(r2), "Masked_RMSE_Phys": float(rmse),
        "Avg_Dice": float(np.mean(dices)), "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0
    }

def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir = Path(f"./models/sta_lstm")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    idx = {"il8":0, "il1":1, "il6":2, "il10":3, "tnf":4, "tgf":5}[cytokine.lower()]
    X = np.load(data_path / "X_lstm.npy").astype(np.float32)[..., idx:idx+1]
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)

    with open(data_path / "metadata.json", "r") as f:
        meta = json.load(f)
    p_min, p_max = meta["scaling"]["min"][idx], meta["scaling"]["max"][idx]

    model = STALSTM(grid)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

    print(f"Training STA-LSTM for {cytokine} on {grid}x{grid}...")
    model.fit(X[:80], Y[:80], epochs=100, batch_size=4, verbose=1)

    Y_p_scaled = model.predict(X, batch_size=2)
    
    Y_p = Y_p_scaled * (p_max - p_min) + p_min
    Y_a = Y * (p_max - p_min) + p_min

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "results": {
            "Interpolation_72_89": calculate_metrics_phys(Y_a[70:88], Y_p[70:88], M[70:88], grid),
            "Extrapolation_82_100": calculate_metrics_phys(Y_a[80:99], Y_p[80:99], M[80:99], grid)
        }
    }

    with open(out_dir / f"res_{suffix}.json", 'w') as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"DONE: {suffix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.grid, args.seed, args.cytokine)
