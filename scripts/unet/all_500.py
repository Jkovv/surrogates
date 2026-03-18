"""
UNet – 500×500 Snellius run
Adaptations vs all.py:
  - batch_size capped at {2, 4}  (skip_0 = batch×500×500×32 = 192 MB @ bs=2)
  - base_filters capped at 32; depth capped at 3
  - X_unet memory-mapped; per-batch loading
  - set_memory_growth enabled
  - Optional dropout in bottleneck
  - --data-dir argument to point at scan-iteration preprocessed folder
"""
import argparse, json, gc, os, time
import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID        = 500
DATA_DIR    = "preprocessed/500x500"   # overridden by --data-dir
RESULTS_DIR = "models/unet"
CYT_MAP     = {"il8": 0, "il10": 3}
FULL_EPOCHS = 200
TUNE_EPOCHS = 20
N_OPTUNA    = 20
PATIENCE    = 20

tf.config.set_visible_devices([], "GPU")  # CPU-only (rome partition)


def build_unet(base_filters, depth, dropout_rate):
    inputs = tf.keras.Input(shape=(GRID, GRID, 22))
    skips  = []

    x = inputs
    f = base_filters
    for d in range(depth):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        f *= 2

    # Bottleneck
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    for d in reversed(range(depth)):
        f //= 2
        x = tf.keras.layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        # Crop skip to match upsampled size (needed when GRID is not divisible by 2^depth)
        x = tf.keras.layers.Lambda(
            lambda t: tf.concat([t[0], t[1][:, :tf.shape(t[0])[1], :tf.shape(t[0])[2], :]], axis=-1)
        )([x, skips[d]])
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Conv2D(1, 1)(x)   # (B, G, G, 1)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def run(cyt_name, seed):
    print(f"\n{'='*60}")
    print(f"UNet 500x500 | cytokine={cyt_name} | seed={seed}")
    print(f"{'='*60}")
    np.random.seed(seed); tf.random.set_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cyt_idx  = CYT_MAP[cyt_name]
    md       = json.load(open(f"{DATA_DIR}/metadata.json"))
    clip     = float(md["clip_max"][cyt_idx])

    Xu_mmap  = np.load(f"{DATA_DIR}/X_unet.npy",          mmap_mode="r")  # (99,2,G,G,11)
    Yt       = np.load(f"{DATA_DIR}/Y_target.npy")                        # (99,G,G,6)
    Yraw     = np.load(f"{DATA_DIR}/Y_raw_phys.npy")[1:, ..., cyt_idx]    # (99,G,G)
    masks    = np.load(f"{DATA_DIR}/Y_masks_spatial.npy")                 # (99,G,G,5)

    def get_batch(idx):
        # Reshape (B,2,G,G,11) → (B,G,G,22) for UNet input
        raw = Xu_mmap[idx].astype(np.float32)                             # (B,2,G,G,11)
        B   = len(idx)
        x   = raw.transpose(0, 2, 3, 1, 4).reshape(B, GRID, GRID, 22)   # (B,G,G,22)
        y   = Yt[idx, :, :, cyt_idx:cyt_idx+1].astype(np.float32)       # (B,G,G,1)
        return x, y

    def objective(trial):
        f   = trial.suggest_categorical("base_filters", [16, 32])
        d   = trial.suggest_categorical("depth",        [2])
        dr  = trial.suggest_categorical("dropout",      [0.0, 0.1, 0.2])
        lr  = trial.suggest_float("lr",         1e-4, 1e-3, log=True)
        bs  = trial.suggest_categorical("batch_size",   [2, 4])

        m   = build_unet(f, d, dr)
        opt = tf.keras.optimizers.Adam(lr)
        tr_idx = np.arange(140)
        for ep in range(TUNE_EPOCHS):
            np.random.shuffle(tr_idx)
            for s in range(0, 140, bs):
                idx      = tr_idx[s:s+bs]
                x_b, y_b = get_batch(idx)
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean((m(x_b, training=True) - y_b)**2)
                grads = tape.gradient(loss, m.trainable_variables)
                opt.apply_gradients(zip(grads, m.trainable_variables))
        vl_idx   = np.arange(140, 160)
        x_v, y_v = get_batch(vl_idx)
        vl = float(tf.reduce_mean((m(x_v, training=False) - y_v)**2).numpy())
        del m; gc.collect(); tf.keras.backend.clear_session()
        return vl

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best = study.best_params
    print(f"Best params: {best}")

    model     = build_unet(best["base_filters"], best["depth"], best["dropout"])
    opt       = tf.keras.optimizers.Adam(best["lr"])
    tr_idx    = np.arange(140)
    best_val, stagnant, best_w = 1e9, 0, None

    t_train_start = time.time()
    for epoch in range(FULL_EPOCHS):
        np.random.shuffle(tr_idx)
        for s in range(0, 140, best["batch_size"]):
            idx      = tr_idx[s:s+best["batch_size"]]
            x_b, y_b = get_batch(idx)
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((model(x_b, training=True) - y_b)**2)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        vl_idx   = np.arange(140, 160)
        x_v, y_v = get_batch(vl_idx)
        vl = float(tf.reduce_mean((model(x_v, training=False) - y_v)**2).numpy())
        if vl < best_val - 1e-6:
            best_val, stagnant, best_w = vl, 0, model.get_weights()
        else:
            stagnant += 1
            if stagnant >= PATIENCE:
                break

    if best_w:
        model.set_weights(best_w)
    train_elapsed = time.time() - t_train_start

    test_idx  = np.arange(160, 199)
    x_te, _   = get_batch(test_idx)
    t_pred_start = time.time()
    pred      = model(x_te, training=False).numpy()[:, :, :, 0]  # (39,G,G)
    pred_elapsed = time.time() - t_pred_start
    pred_phys = (pred + 1.0) / 2.0 * clip
    pred_phys = np.clip(pred_phys, 0, None)
    y_true    = Yraw[160:199]

    r2   = float(r2_score(y_true.flatten(), pred_phys.flatten()))
    rmse = float(np.sqrt(np.mean((y_true - pred_phys)**2)))
    ssim_v = float(np.mean([ssim(y_true[t], pred_phys[t], data_range=clip)
                             for t in range(39) if y_true[t].std() > 1e-12]))
    metrics = {"Global_R2": r2, "Unmasked_RMSE": rmse, "SSIM": ssim_v,
               "cytokine": cyt_name, "seed": seed, "grid": 500,
               "train_time_seconds": round(train_elapsed, 2),
               "pred_time_seconds":  round(pred_elapsed, 4),
               "best_params": best, "model": "unet"}
    out_path = f"{RESULTS_DIR}/res_{cyt_name}_500_{seed}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
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
