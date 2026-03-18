"""
STA-LSTM – 500×500 Snellius run
Adaptations vs all.py:
  - X_lstm memory-mapped (never fully loaded into RAM)
  - Custom training loop replaces model.fit() to enable mmap + small batches
  - batch_size capped at {2, 4}
  - filters capped at {32, 64}
  - lstm_units capped at {64, 128}
  - set_memory_growth enabled
  - --data-dir argument to point at scan-iteration preprocessed folder
"""
import argparse, json, gc, os, random, time
import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID        = 500
DATA_DIR    = "preprocessed/500x500"   # overridden by --data-dir
RESULTS_DIR = "models/sta_lstm"
CYT_MAP     = {"il8": 0, "il10": 3}
N_OPTUNA    = 20
TUNE_EPOCHS = 20
FULL_EPOCHS = 200
PATIENCE    = 20

tf.config.set_visible_devices([], "GPU")  # CPU-only (rome partition)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = tf.keras.layers.Conv2D(
            1, kernel_size=1, padding="same", activation="sigmoid"
        )

    def call(self, x):
        return x * self.attn_conv(x)


class STALSTM(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=128):
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
        self.fc   = tf.keras.layers.Dense(self.latent_size * self.latent_size * filters, activation="relu")
        self.reshape_latent = tf.keras.layers.Reshape((self.latent_size, self.latent_size, filters))
        self.deconv1   = tf.keras.layers.Conv2DTranspose(filters // 2, 3, strides=2, padding="same", activation="relu")
        self.deconv2   = tf.keras.layers.Conv2DTranspose(filters // 4, 3, strides=2, padding="same", activation="relu")
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

def calculate_metrics(y_true, y_pred, masks, clip_max):
    min_t = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:min_t]
    yp = np.maximum(y_pred[:min_t], 0.0)
    ms = np.max(masks[:min_t], axis=-1, keepdims=True)

    sq_diff = np.square(yt - yp)
    rmse    = float(np.sqrt(np.sum(sq_diff * ms) / (np.sum(ms) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))
    r2 = float(r2_score(yt.flatten(), yp.flatten()))

    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices, n_empty, z_corrs, ssims_v, n_ssim_skip = [], 0, [], [], 0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    for t in range(min_t):
        gt = yt[t, :, :, 0]; pr = yp[t, :, :, 0]
        gb = (gt > dice_thr).astype(float); pb = (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) == 0:
            n_empty += 1
        else:
            dices.append((2.0 * np.sum(gb * pb)) / (np.sum(gb) + np.sum(pb) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))
        if float(np.max(gt) - np.min(gt)) > 1e-12:
            ssims_v.append(float(ssim(gt, pr, data_range=fixed_dr)))
        else:
            n_ssim_skip += 1

    return {
        "Global_R2":           r2,
        "Masked_RMSE":         rmse,
        "Unmasked_RMSE":       unmasked_rmse,
        "Avg_Dice":            float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped":  n_empty,
        "Spatial_Correlation": float(np.tanh(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM":                float(np.mean(ssims_v)) if ssims_v else 0.0,
        "SSIM_Skipped_Frames": n_ssim_skip,
    }

def denormalize(scaled, clip_max):
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max


def get_batch(Xm, Yt_arr, cyt_idx, indices):
    """Load a batch from mmap'd X_lstm. Returns (B,2,G,G,11) and (B,G,G,1)."""
    x = Xm[indices].astype(np.float32)                             # (B,2,G,G,11)
    y = Yt_arr[indices, :, :, cyt_idx:cyt_idx+1].astype(np.float32)  # (B,G,G,1)
    return x, y


N_TRAIN = 140   # 0:140
N_VAL   = 160   # 140:160
# test  = 160:198  (39 samples, reported in 3 windows of 13)


def train_loop(model, Xm, Yt_arr, cyt_idx, lr, bs, epochs, patience, vl_idx):
    opt = tf.keras.optimizers.Adam(lr)
    tr_idx = np.arange(N_TRAIN)
    best_val, stagnant, best_w = 1e9, 0, None

    for epoch in range(epochs):
        np.random.shuffle(tr_idx)
        for s in range(0, N_TRAIN, bs):
            idx      = tr_idx[s:s+bs]
            x_b, y_b = get_batch(Xm, Yt_arr, cyt_idx, idx)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((model(x_b, training=True) - y_b)**2)
            opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                    model.trainable_variables))

        x_v, y_v = get_batch(Xm, Yt_arr, cyt_idx, vl_idx)
        vl = float(tf.reduce_mean((model(x_v, training=False) - y_v)**2).numpy())
        if vl < best_val - 1e-6:
            best_val, stagnant, best_w = vl, 0, model.get_weights()
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    if best_w:
        model.set_weights(best_w)
    return best_val


def run(cyt_name, seed):
    print(f"\n{'='*60}")
    print(f"STA-LSTM 500x500 | cytokine={cyt_name} | seed={seed}")
    print(f"{'='*60}")
    set_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cyt_idx = CYT_MAP[cyt_name]
    md      = json.load(open(f"{DATA_DIR}/metadata.json"))
    clip    = float(md["clip_max"][cyt_idx])

    Xm = np.load(f"{DATA_DIR}/X_lstm.npy",          mmap_mode="r")  # (N,2,G,G,11)
    Yt = np.load(f"{DATA_DIR}/Y_target.npy")                        # (N,G,G,6)
    M  = np.load(f"{DATA_DIR}/Y_masks_spatial.npy")                 # (N,G,G,5)
    vl_idx = np.arange(N_TRAIN, N_VAL)

    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        f   = trial.suggest_categorical("filters",    [32, 64])
        lu  = trial.suggest_categorical("lstm_units", [64, 128])
        lr  = trial.suggest_float("lr",         1e-5, 1e-3, log=True)
        bs  = trial.suggest_categorical("batch_size", [2, 4])

        m = STALSTM(grid_size=GRID, filters=f, lstm_units=lu)
        vl = train_loop(m, Xm, Yt, cyt_idx, lr, bs, TUNE_EPOCHS, patience=5, vl_idx=vl_idx)
        del m; gc.collect(); tf.keras.backend.clear_session()
        return vl

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best = study.best_params
    print(f"Best params: {best}")

    set_seed(seed); tf.keras.backend.clear_session()
    model = STALSTM(grid_size=GRID, filters=best["filters"], lstm_units=best["lstm_units"])

    t_start = time.time()
    train_loop(model, Xm, Yt, cyt_idx, best["lr"], best["batch_size"],
               FULL_EPOCHS, PATIENCE, vl_idx)
    train_elapsed = time.time() - t_start

    # Predict test set (160:198) in small batches to avoid OOM
    n_total = Xm.shape[0]   # 199 after full preprocessing
    te_idx  = np.arange(N_VAL, n_total)   # 160:199 = 39 samples
    all_preds = []
    t_pred_start = time.time()
    for s in range(0, len(te_idx), 4):
        idx    = te_idx[s:s+4]
        x_b, _ = get_batch(Xm, Yt, cyt_idx, idx)
        all_preds.append(model(x_b, training=False).numpy())
    pred_elapsed = time.time() - t_pred_start
    Yp_scaled = np.concatenate(all_preds, axis=0)          # (39,G,G,1)

    Yp_phys = denormalize(Yp_scaled, clip)
    Ya_phys = denormalize(Yt[te_idx, ..., cyt_idx:cyt_idx+1], clip)
    M_test  = M[te_idx]

    # 3 equal windows of 13 samples each over the 39-sample test set
    w = len(te_idx) // 3   # 13
    results = {
        "grid": GRID, "seed": seed, "cytokine": cyt_name,
        "best_params": best,
        "train_time_seconds": round(train_elapsed, 2),
        "pred_time_seconds":  round(pred_elapsed, 4),
        "split": {"n_train": N_TRAIN, "n_val": N_VAL - N_TRAIN, "n_test": len(te_idx)},
        "results": {
            "Window1_t162_t174": calculate_metrics(
                Ya_phys[0*w:1*w], Yp_phys[0*w:1*w], M_test[0*w:1*w], clip
            ),
            "Window2_t175_t187": calculate_metrics(
                Ya_phys[1*w:2*w], Yp_phys[1*w:2*w], M_test[1*w:2*w], clip
            ),
            "Window3_t188_t200": calculate_metrics(
                Ya_phys[2*w:],    Yp_phys[2*w:],    M_test[2*w:],    clip
            ),
        },
    }
    out_path = f"{RESULTS_DIR}/res_{cyt_name}_500_{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved → {out_path}")
    del model; gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytokine", choices=["il8","il10"], required=True)
    parser.add_argument("--seed",     type=int,               required=True)
    parser.add_argument("--data-dir", default="preprocessed/500x500",
                        dest="data_dir",
                        help="Path to preprocessed/500x500 directory")
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    run(args.cytokine, args.seed)
