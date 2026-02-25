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

class GatedTrunk(tf.keras.layers.Layer):
    def __init__(self, hidden: int = 128, p: int = 128):
        super().__init__()
        self.hidden = hidden
        self.p      = p

    def build(self, input_shape):
        self.U = tf.keras.layers.Dense(self.hidden, activation="tanh")
        self.V = tf.keras.layers.Dense(self.hidden, activation="tanh")

        # main hidden layers
        self.W1  = tf.keras.layers.Dense(self.hidden, activation="tanh")
        self.W2  = tf.keras.layers.Dense(self.hidden, activation="tanh")
        self.out = tf.keras.layers.Dense(self.p)   # linear FC output

        super().build(input_shape)

    def call(self, x):
        u = self.U(x)
        v = self.V(x)

        # Hidden layer 1 + Hadamard gate
        h = self.W1(x)
        h = h * u + (1.0 - h) * v

        # Hidden layer 2 + Hadamard gate
        h = self.W2(h)
        h = h * u + (1.0 - h) * v

        return self.out(h) # (batch, n_points, p)


# DeepONet 

class DeepONet(tf.keras.Model):
    def __init__(self, grid_size: int, p: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.p         = p

        # Branch 
        self.branch = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.GlobalAveragePooling2D()
            ),
            tf.keras.layers.LSTM(128, return_sequences=False),
            tf.keras.layers.Activation("relu"),  
            tf.keras.layers.Dense(p),          
        ], name="branch")

        # Trunk 
        self.trunk = GatedTrunk(hidden=128, p=p)

        self.bias = self.add_weight(
            shape=(1,), initializer="zeros", trainable=True, name="bias"
        )

    def call(self, inputs):
        x_branch, x_trunk = inputs
     
        b_out = self.branch(x_branch)  
        t_out = self.trunk(x_trunk)  

        res = tf.einsum("bp,bnp->bn", b_out, t_out) + self.bias  # (batch, G*G)

        return tf.reshape(res, [-1, self.grid_size, self.grid_size, 1])


# metrics 
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      masks: np.ndarray) -> dict:
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    y_t   = y_true[:min_t]
    y_p   = np.maximum(y_pred[:min_t], 0.0)   # concentrations are non-negative

    m_s = np.max(masks[:min_t], axis=-1, keepdims=True)   # (T, G, G, 1)

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

        # spatial pearson corr
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
    """
    preprocessing: u_scaled = clip(u, 0, clip_max) / clip_max * 2 - 1
    inverse:       u_phys   = (u_scaled + 1) / 2 * clip_max
    """
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max


# pipeline
def run_pipeline(grid: int, seed: int, cytokine: str):
    set_seed(seed)

    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx       = cyt_names.index(cytokine.lower())

    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir   = Path("./models/deeponet_nofourier")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_b = np.load(data_path / "X_branch.npy").astype(np.float32)
    X_t = np.load(data_path / "X_trunk.npy").astype(np.float32)
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)

    with open(data_path / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    model = DeepONet(grid_size=grid, p=128)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="mse",
    )

    print(f"Training DeepONet [{cytokine.upper()}] on {grid}x{grid}...")
    model.fit(
        [X_b[:80], X_t[:80]],
        Y[:80],
        validation_data=([X_b[80:90], X_t[80:90]], Y[80:90]),
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

    Y_p_scaled = model.predict([X_b, X_t], batch_size=2) # [-1, 1]
    Y_p_phys   = denormalize(Y_p_scaled, clip_max) # physical units
    Y_a_phys   = denormalize(Y,          clip_max)

    # eval
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
    parser.add_argument("--grid",     type=int,  default=None)
    parser.add_argument("--cytokine", type=str,  required=True)
    parser.add_argument("--seed",     type=int,  default=42)
    args = parser.parse_args()

    if args.grid:
        run_pipeline(args.grid, args.seed, args.cytokine)
    else:
        for d in sorted(Path("./preprocessed").iterdir()):
            if d.is_dir():
                grid_size = int(d.name.split("x")[0])
                run_pipeline(grid_size, args.seed, args.cytokine)
