import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size: int, lstm_units: int = 128):
        super().__init__()
        self.grid_size  = grid_size
        self.lstm_units = lstm_units

        self.flatten_spatial = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        )

        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        self.relu = tf.keras.layers.Activation("relu")
        self.fc = tf.keras.layers.Dense(grid_size * grid_size, activation="linear")
        self.reshape = tf.keras.layers.Reshape((grid_size, grid_size, 1))

    def call(self, x):
        # x: (batch, 2, G, G, 11)
        h = self.flatten_spatial(x)   # (batch, 2, G*G*11)
        h = self.lstm(h)              # (batch, lstm_units)
        h = self.relu(h)              # (batch, lstm_units)
        h = self.fc(h)                # (batch, G*G)
        return self.reshape(h)        # (batch, G, G, 1)


# metrics
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      masks: np.ndarray) -> dict:
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    y_t   = y_true[:min_t]
    y_p   = np.maximum(y_pred[:min_t], 0.0)
    m_s   = np.max(masks[:min_t], axis=-1, keepdims=True)

    # masked RMSE
    sq_diff = np.square(y_t - y_p) * m_s
    rmse    = float(np.sqrt(np.sum(sq_diff) / (np.sum(m_s) + 1e-12)))

    # global R²
    r2 = float(r2_score(y_t.flatten(), y_p.flatten()))

    dices, corrs, ssims = [], [], []
    for t in range(min_t):
        gt = y_t[t, :, :, 0]
        pr = y_p[t, :, :, 0]

        # dice
        thresh = 0.05 * float(np.max(gt)) if np.max(gt) > 0 else 1e-9
        g_b    = (gt > thresh).astype(float)
        p_b    = (pr > thresh).astype(float)
        dices.append(
            (2.0 * np.sum(g_b * p_b) + 1e-6) / (np.sum(g_b) + np.sum(p_b) + 1e-6)
        )

        # Spatial correlation
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            corrs.append(float(pearsonr(gt.flatten(), pr.flatten())[0]))

        # SSIM
        data_range = float(np.max(gt) - np.min(gt))
        if data_range > 1e-12:
            ssims.append(float(ssim(gt, pr, data_range=data_range)))

    return {
        "Global_R2":           r2,
        "Masked_RMSE":         rmse,
        "Avg_Dice":            float(np.mean(dices)),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0,
        "SSIM":                float(np.mean(ssims))  if ssims  else 0.0,
    }

def denormalize(scaled: np.ndarray, clip_max: float) -> np.ndarray:
    """u_phys = (u_scaled + 1) / 2 * clip_max"""
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max


def run_pipeline(grid: int, seed: int, cytokine: str):
    set_seed(seed)

    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx       = cyt_names.index(cytokine.lower())

    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir   = Path("./models/sta_lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_path / "X_lstm.npy").astype(np.float32)
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)

    with open(data_path / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    model = STALSTM(grid_size=grid, lstm_units=128)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
    )

    print(f"Training STA-LSTM [{cytokine.upper()}] on {grid}x{grid}...")
    model.fit(
        X[:80], Y[:80],
        validation_data=(X[80:90], Y[80:90]),
        epochs=200,
        batch_size=4,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
            ),
        ],
    )

    # Predict and denormalise
    Y_p_scaled = model.predict(X, batch_size=2)
    Y_p_phys   = denormalize(Y_p_scaled, clip_max)
    Y_a_phys   = denormalize(Y,          clip_max)

    suffix  = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid":     grid,
        "seed":     seed,
        "cytokine": cytokine,
        "results": {
            "Interpolation_72_89":  calculate_metrics(
                Y_a_phys[70:88], Y_p_phys[70:88], M[70:88]
            ),
            "Extrapolation_82_100": calculate_metrics(
                Y_a_phys[80:99], Y_p_phys[80:99], M[80:99]
            ),
        },
    }

    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"DONE: {suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid",     type=int, default=None)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    if args.grid:
        run_pipeline(args.grid, args.seed, args.cytokine)
    else:
        for d in sorted(Path("./preprocessed").iterdir()):
            if d.is_dir():
                grid_size = int(d.name.split("x")[0])
                run_pipeline(grid_size, args.seed, args.cytokine)
