import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 20
TUNE_EPOCHS = 20
FULL_EPOCHS = 200

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, padding="same", activation="sigmoid"
        )

    def call(self, x):
        attn_map = self.attn_conv(x)      
        return x * attn_map              

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size: int, filters: int = 64,
                 lstm_units: int = 128):
        super().__init__()
        self.grid_size   = grid_size
        self.filters     = filters
        self.lstm_units  = lstm_units
        self.latent_size = max(grid_size // 4, 8) 

        # encoder 
        self.enc = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters, 3, strides=2, padding="same", activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters, 3, strides=2, padding="same", activation="relu"
                ),
            ]), name="encoder"
        )

        self.spatial_attn = tf.keras.layers.TimeDistributed(
            SpatialAttention(), name="spatial_attention"
        )

        self.gap  = tf.keras.layers.TimeDistributed(
            tf.keras.layers.GlobalAveragePooling2D(), name="gap"
        )
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False,
                                         name="lstm")
        self.relu = tf.keras.layers.Activation("relu")
        self.fc = tf.keras.layers.Dense(
            self.latent_size * self.latent_size * filters, activation="relu"
        )
        self.reshape_latent = tf.keras.layers.Reshape(
            (self.latent_size, self.latent_size, filters)
        )

        # decoder
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            filters // 2, 3, strides=2, padding="same", activation="relu"
        )
        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            filters // 4, 3, strides=2, padding="same", activation="relu"
        )
        self.out_conv  = tf.keras.layers.Conv2D(1, 3, padding="same",
                                                activation="linear")
        self.out_resize = tf.keras.layers.Resizing(grid_size, grid_size)

    def call(self, x):
        h = self.enc(x)              
        h = self.spatial_attn(h)     
        h = self.gap(h)               
        h = self.lstm(h)              
        h = self.relu(h)
        h = self.fc(h)                
        h = self.reshape_latent(h)    
        h = self.deconv1(h)           
        h = self.deconv2(h)           
        h = self.out_conv(h)          
        return self.out_resize(h)     

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      masks: np.ndarray) -> dict:
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    y_t   = y_true[:min_t]
    y_p   = np.maximum(y_pred[:min_t], 0.0)
    m_s   = np.max(masks[:min_t], axis=-1, keepdims=True)

    sq_diff = np.square(y_t - y_p) * m_s
    rmse    = float(np.sqrt(np.sum(sq_diff) / (np.sum(m_s) + 1e-12)))
    r2      = float(r2_score(y_t.flatten(), y_p.flatten()))

    dices, corrs, ssims = [], [], []
    for t in range(min_t):
        gt = y_t[t, :, :, 0]
        pr = y_p[t, :, :, 0]

        thresh = 0.05 * float(np.max(gt)) if np.max(gt) > 0 else 1e-9
        g_b    = (gt > thresh).astype(float)
        p_b    = (pr > thresh).astype(float)
        dices.append(
            (2.0 * np.sum(g_b * p_b) + 1e-6) / (np.sum(g_b) + np.sum(p_b) + 1e-6)
        )
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            corrs.append(float(pearsonr(gt.flatten(), pr.flatten())[0]))

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
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max

def make_objective(X_train, Y_train, X_val, Y_val, grid_size, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()

        filters    = trial.suggest_categorical("filters",    [32, 64])
        lstm_units = trial.suggest_categorical("lstm_units", [64, 128, 256])
        lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])

        model = STALSTM(grid_size=grid_size, filters=filters,
                        lstm_units=lstm_units)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=TUNE_EPOCHS,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
        )
        return float(min(history.history["val_loss"]))

    return objective

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

    X_train, Y_train = X[:70],   Y[:70]
    X_val,   Y_val   = X[70:80], Y[70:80]

    # optuna 
    print(f"\nOptuna [{cytokine.upper()}] {grid}x{grid} — "
          f"{N_TRIALS} trials × {TUNE_EPOCHS} epochs...")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        make_objective(X_train, Y_train, X_val, Y_val, grid, seed),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"  Best: {best}  |  val_loss = {study.best_value:.6f}")

    # final train 
    tf.keras.backend.clear_session()
    set_seed(seed)

    model = STALSTM(
        grid_size=grid,
        filters=best["filters"],
        lstm_units=best["lstm_units"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(best["learning_rate"]),
        loss="mse",
    )

    print(f"Final training [{cytokine.upper()}] {grid}x{grid}...")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=FULL_EPOCHS,
        batch_size=best["batch_size"],
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

    # eval 
    Y_p_scaled = model.predict(X, batch_size=2)
    Y_p_phys   = denormalize(Y_p_scaled, clip_max)
    Y_a_phys   = denormalize(Y,          clip_max)

    suffix  = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid":                 grid,
        "seed":                 seed,
        "cytokine":             cytokine,
        "best_params":          best,
        "optuna_best_val_loss": float(study.best_value),
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
