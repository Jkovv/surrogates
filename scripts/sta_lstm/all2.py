import os
import json
import argparse
import random
import time
import sys
import subprocess
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
        return x * self.attn_conv(x)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size: int, filters: int = 64,
                 lstm_units: int = 128):
        super().__init__()
        self.grid_size   = grid_size
        self.filters     = filters
        self.lstm_units  = lstm_units
        self.latent_size = max(grid_size // 4, 8)
        self.enc = tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
            ]), name="encoder"
        )
        self.spatial_attn = tf.keras.layers.TimeDistributed(SpatialAttention(), name="spatial_attention")
        self.gap  = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), name="gap")
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False, name="lstm")
        self.relu = tf.keras.layers.Activation("relu")
        self.fc = tf.keras.layers.Dense(self.latent_size * self.latent_size * filters, activation="relu")
        self.reshape_latent = tf.keras.layers.Reshape((self.latent_size, self.latent_size, filters))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters // 2, 3, strides=2, padding="same", activation="relu")
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters // 4, 3, strides=2, padding="same", activation="relu")
        self.out_conv  = tf.keras.layers.Conv2D(1, 3, padding="same", activation="linear")
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


def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def _inv_fisher_z(z):
    return float(np.tanh(z))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    y_t = y_true[:min_t]; y_p = np.maximum(y_pred[:min_t], 0.0)
    m_s = np.max(masks[:min_t], axis=-1, keepdims=True)
    sq_diff = np.square(y_t - y_p)
    rmse = float(np.sqrt(np.sum(sq_diff * m_s) / (np.sum(m_s) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))
    r2 = float(r2_score(y_t.flatten(), y_p.flatten()))
    per_t_r2 = []
    for t in range(min_t):
        gt_f = y_t[t].flatten(); pr_f = y_p[t].flatten()
        per_t_r2.append(float(r2_score(gt_f, pr_f)) if np.std(gt_f) > 1e-12 else np.nan)
    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices, n_empty = [], 0
    z_corrs = []
    ssims_v, n_ssim_skip = [], 0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    for t in range(min_t):
        gt = y_t[t, :, :, 0]; pr = y_p[t, :, :, 0]
        g_b = (gt > dice_thr).astype(float); p_b = (pr > dice_thr).astype(float)
        if np.sum(g_b) + np.sum(p_b) == 0:
            n_empty += 1
        else:
            dices.append((2.0 * np.sum(g_b * p_b)) / (np.sum(g_b) + np.sum(p_b) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))
        dr = float(np.max(gt) - np.min(gt))
        if dr > 1e-12:
            ssims_v.append(float(ssim(gt, pr, data_range=fixed_dr)))
        else:
            n_ssim_skip += 1
    return {
        "Global_R2": r2, "Per_Timestep_R2": per_t_r2,
        "Masked_RMSE": rmse, "Unmasked_RMSE": unmasked_rmse,
        "Avg_Dice": float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped": n_empty,
        "Spatial_Correlation": _inv_fisher_z(float(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM": float(np.mean(ssims_v)) if ssims_v else 0.0,
        "SSIM_Skipped_Frames": n_ssim_skip,
    }

def denormalize(scaled, clip_max):
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max

def make_objective(X_train, Y_train, X_val, Y_val, grid_size, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        filters    = trial.suggest_categorical("filters",    [32, 64])
        lstm_units = trial.suggest_categorical("lstm_units", [64, 128])
        lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        model = STALSTM(grid_size=grid_size, filters=filters, lstm_units=lstm_units)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
        history = model.fit(
            X_train, Y_train, validation_data=(X_val, Y_val),
            epochs=TUNE_EPOCHS, batch_size=batch_size, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )],
        )
        return float(min(history.history["val_loss"]))
    return objective


def _predict_subprocess(script_path, data_path, weights_path, pred_path,
                        grid, filters, lstm_units, cyt_idx):
    code = f"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.attn_conv = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")
    def call(self, x):
        return x * self.attn_conv(x)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=128):
        super().__init__()
        ls = max(grid_size // 4, 8)
        self.enc = tf.keras.layers.TimeDistributed(tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
        ]), name="encoder")
        self.spatial_attn = tf.keras.layers.TimeDistributed(SpatialAttention(), name="spatial_attention")
        self.gap = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), name="gap")
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False, name="lstm")
        self.relu = tf.keras.layers.Activation("relu")
        self.fc = tf.keras.layers.Dense(ls * ls * filters, activation="relu")
        self.reshape_latent = tf.keras.layers.Reshape((ls, ls, filters))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters//2, 3, strides=2, padding="same", activation="relu")
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters//4, 3, strides=2, padding="same", activation="relu")
        self.out_conv = tf.keras.layers.Conv2D(1, 3, padding="same", activation="linear")
        self.out_resize = tf.keras.layers.Resizing(grid_size, grid_size)
    def call(self, x):
        h = self.enc(x); h = self.spatial_attn(h); h = self.gap(h)
        h = self.lstm(h); h = self.relu(h); h = self.fc(h); h = self.reshape_latent(h)
        return self.out_resize(self.out_conv(self.deconv2(self.deconv1(h))))

X = np.load("{data_path}/X_lstm.npy").astype(np.float32)
model = STALSTM(grid_size={grid}, filters={filters}, lstm_units={lstm_units})
model.build(X[:1].shape)
model.load_weights("{weights_path}")
Y_p = np.concatenate([model(X[i:i+1], training=False).numpy() for i in range(len(X))], axis=0)
np.save("{pred_path}", Y_p)
print("PREDICT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=3600,
    )
    if "PREDICT_OK" not in result.stdout:
        print("Subprocess STDERR:", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError("Prediction subprocess failed")


def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx = cyt_names.index(cytokine.lower())

    data_path = Path(f"./preprocessed_200h/{grid}x{grid}")
    out_dir   = Path("./models/200hrs/sta_lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_path / "X_lstm.npy").astype(np.float32)
    Y = np.load(data_path / "Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M = np.load(data_path / "Y_masks_spatial.npy").astype(np.float32)

    with open(data_path / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    X_train, Y_train = X[:140],   Y[:140]
    X_val,   Y_val   = X[140:160], Y[140:160]

    if seed == 42:
        print(f"\nOptuna [{cytokine.upper()}] {grid}x{grid} — "
              f"{N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            make_objective(X_train, Y_train, X_val, Y_val, grid, 42),
            n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
        )
        best = study.best_params
        optuna_val = float(study.best_value)
        print(f"  Best: {best}  |  val_loss = {optuna_val:.6f}")
    else:
        ref_path = out_dir / f"res_{cytokine}_{grid}_42.json"
        print(f"  Loading HP from {ref_path.name}")
        with open(ref_path) as f:
            ref = json.load(f)
        best = ref["best_params"]
        optuna_val = ref["optuna_best_val_loss"]

    tf.keras.backend.clear_session()
    set_seed(seed)

    t_start = time.time()

    model = STALSTM(grid_size=grid, filters=best["filters"], lstm_units=best["lstm_units"])
    model.compile(optimizer=tf.keras.optimizers.Adam(best["learning_rate"]), loss="mse")

    print(f"Final training [{cytokine.upper()}] {grid}x{grid}...")
    model.fit(
        X_train, Y_train, validation_data=(X_val, Y_val),
        epochs=FULL_EPOCHS, batch_size=best["batch_size"], verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ],
    )

    train_elapsed = time.time() - t_start
    suffix = f"{cytokine}_{grid}_{seed}"
    weights_path = out_dir / f"weights_{suffix}.weights.h5"
    model.save_weights(str(weights_path))

    # Predict in subprocess on CPU (fresh process, no GPU memory issues)
    pred_path = str(out_dir / f"_pred_{suffix}.npy")
    print("  Predicting in subprocess (CPU)...")
    t_pred = time.time()
    _predict_subprocess(
        script_path=__file__,
        data_path=str(data_path),
        weights_path=str(weights_path),
        pred_path=pred_path,
        grid=grid,
        filters=best["filters"],
        lstm_units=best["lstm_units"],
        cyt_idx=idx,
    )
    pred_elapsed = time.time() - t_pred

    Y_p_scaled = np.load(pred_path)
    os.remove(pred_path)

    Y_p_phys = denormalize(Y_p_scaled, clip_max)
    Y_a_phys = denormalize(Y, clip_max)

    print(f"  Train: {train_elapsed:.1f}s | Pred: {pred_elapsed:.1f}s")

    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params": best,
        "optuna_best_val_loss": optuna_val,
        "train_time_seconds": round(train_elapsed, 2),
        "pred_time_seconds":  round(pred_elapsed, 2),
        "results": {
            "Near_Horizon_t162_t181": calculate_metrics(
                Y_a_phys[160:180], Y_p_phys[160:180], M[160:180], clip_max),
            "Far_Horizon_t182_t200": calculate_metrics(
                Y_a_phys[180:199], Y_p_phys[180:199], M[180:199], clip_max),
        },
    }

    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"DONE: {suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    grid_dir = next(Path("./preprocessed_200h").iterdir())
    grid = int(grid_dir.name.split("x")[0])
    run_pipeline(grid, args.seed, args.cytokine)