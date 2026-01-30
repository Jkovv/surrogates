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

CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')
        self.attn = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')

    def call(self, inputs):
        x = self.conv(inputs)
        a = self.attn(x)
        return inputs * a

class STALSTMSingle(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=64):
        super().__init__()
        self.grid_size = grid_size
        
        self.time_dist = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu'),
                SpatialAttention(filters),
                tf.keras.layers.GlobalAveragePooling2D()
            ])
        )
        
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        
        base_size = grid_size // 5 
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(base_size * base_size * filters // 2, activation='relu'),
            tf.keras.layers.Reshape((base_size, base_size, filters // 2)),
            tf.keras.layers.UpSampling2D(size=(5, 5), interpolation='bilinear'),
            tf.keras.layers.Conv2D(filters // 4, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, 3, padding='same', activation='softplus')
        ])

    def call(self, inputs):
        x = self.time_dist(inputs)
        x = self.lstm(x)
        return self.decoder(x)

# metrics
def compute_dice_coefficient(y_true, y_pred, smooth=1e-6):
    threshold = 0.1 * np.max(y_true)
    if threshold == 0: threshold = 1e-7
    
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin)
    
    return (2. * intersection + smooth) / (union + smooth)

def calculate_metrics(y_true, y_pred, masks):
    spatial_mask = np.max(masks, axis=-1, keepdims=True) 
    
    # masked rmse
    sq_diff = np.square(y_true - y_pred) * spatial_mask
    m_rmse = np.sqrt(np.sum(sq_diff) / (np.sum(spatial_mask) + 1e-7))
    
    # r2
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    dices, corrs, ssims = [], [], []
    for t in range(y_true.shape[0]):
        gt, pr = y_true[t,:,:,0], y_pred[t,:,:,0]
        
        # dice
        dices.append(compute_dice_coefficient(gt, pr))
        
        ssim
        data_range = max(gt.max() - gt.min(), 1e-7)
        ssims.append(ssim(gt, pr, data_range=data_range))
        
        # spatial corr
        if np.std(gt) > 1e-9 and np.std(pr) > 1e-9:
            corrs.append(pearsonr(gt.flatten(), pr.flatten())[0])
            
    sig_gt = np.mean(y_true * spatial_mask, axis=(1, 2))
    sig_pr = np.mean(y_pred * spatial_mask, axis=(1, 2))
    corr_t = np.correlate(sig_gt.flatten(), sig_pr.flatten(), mode='full')
    lag = np.arange(-len(sig_gt)+1, len(sig_gt))[np.argmax(corr_t)]

    return {
        "Global_R2": float(r2), 
        "Masked_RMSE": float(m_rmse),
        "Avg_Dice": float(np.mean(dices)),      
        "Avg_SSIM": float(np.mean(ssims)),      
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0,
        "Peak_Lag": int(lag)
    }

def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir = Path(f"./models/sta_lstm_single")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{cytokine}_grid{grid}_seed{seed}"
    idx = CYTOKINE_MAP[cytokine]

    X_all = np.load(data_path / "X_lstm.npy").astype(np.float32)[..., idx:idx+1]
    Y_all = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M_all = np.load(data_path / "Y_masks.npy").astype(np.float32)

    n = len(X_all)
    t_end, v_end = int(0.7 * n), int(0.8 * n)

    def objective(trial):
        tf.keras.backend.clear_session()
        f = trial.suggest_categorical("filters", [32, 64, 128])
        u = trial.suggest_categorical("lstm_units", [64, 128])
        model = STALSTMSingle(grid, filters=f, lstm_units=u)
        model.compile(optimizer=tf.keras.optimizers.Adam(trial.suggest_float("lr", 1e-4, 1e-3, log=True)), loss="mse")
        model.fit(X_all[:t_end], Y_all[:t_end], epochs=40, batch_size=4, verbose=0)
        return model.evaluate(X_all[t_end:v_end], Y_all[t_end:v_end], verbose=0)

    print(f"optimizing for {cytokine}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10) 
    
    best = study.best_params
    final_model = STALSTMSingle(grid, filters=best["filters"], lstm_units=best["lstm_units"])
    final_model.compile(optimizer=tf.keras.optimizers.Adam(best["lr"]), loss="mse")
    
    print(f">>> Training Best STA-LSTM Model...")
    final_model.fit(X_all[:v_end], Y_all[:v_end], epochs=500, batch_size=4, verbose=1)
    
    final_model.save_weights(out_dir / f"weights_{suffix}.weights.h5")

    Y_pred = final_model.predict(X_all, verbose=0)
    
    res = {
        "params": best, "seed": seed, "grid": grid, "cytokine": cytokine,
        "results": {
            "Window_82_100": calculate_metrics(Y_all[80:99], Y_pred[80:99], M_all[80:99]),
            "Window_72_89": calculate_metrics(Y_all[70:88], Y_pred[70:88], M_all[70:88])
        }
    }
    
    with open(out_dir / f"results_{suffix}.json", 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Success! JSON with DICE & SSIM saved: results_{suffix}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.grid, args.seed, args.cytokine.lower())
