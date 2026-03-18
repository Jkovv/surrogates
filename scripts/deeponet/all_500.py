"""
DeepONet – 500×500 Snellius run
Adaptations vs all.py:
  - X_branch memory-mapped (never fully loaded into RAM)
  - Branch stats pre-computed once and cached as (99,7)
  - chunk_size capped at 1024; Optuna p <= 128
  - set_memory_growth enabled
  - --data-dir argument to point at scan-iteration preprocessed folder
"""
import argparse, json, gc, os, sys, time
import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── constants ──────────────────────────────────────────────────────────────
GRID         = 500
G2           = GRID * GRID
DATA_DIR     = "preprocessed/500x500"   # overridden by --data-dir
RESULTS_DIR  = "models/deeponet_h"
CYT_MAP      = {"il8": 0, "il10": 3}
TRAIN_SL     = slice(0, 140)
VAL_SL       = slice(140, 160)
TEST_SL      = slice(160, 199)
N_OPTUNA     = 20
TUNE_EPOCHS  = 30
FULL_EPOCHS  = 400
PATIENCE     = 40
LR_PATIENCE  = 15
CHUNK_SIZE   = 1024          # max spatial points per forward pass on GPU

# ── GPU setup ─────────────────────────────────────────────────────────────
tf.config.set_visible_devices([], "GPU")  # CPU-only (rome partition)

# ── helpers ────────────────────────────────────────────────────────────────
def load_data(cyt_idx):
    """Load all arrays needed for DeepONet. X_branch is memory-mapped."""
    md   = json.load(open(f"{DATA_DIR}/metadata.json"))
    clip = float(md["clip_max"][cyt_idx])

    # Memory-map the large array — never fully materialised
    Xb   = np.load(f"{DATA_DIR}/X_branch.npy", mmap_mode="r")  # (99,2,G,G,11)
    Xt   = np.load(f"{DATA_DIR}/X_trunk.npy")                   # (99,G²,3)
    Yt   = np.load(f"{DATA_DIR}/Y_target.npy")[..., cyt_idx]    # (99,G,G)
    Yraw = np.load(f"{DATA_DIR}/Y_raw_phys.npy")[1:, ..., cyt_idx]  # (99,G,G)
    masks = np.load(f"{DATA_DIR}/Y_masks_spatial.npy")          # (99,G,G,5)
    t_norm = np.linspace(0, 1, 199, dtype=np.float32)

    # Pre-compute branch stats once (shape: 199×7)  — avoids loading 5.24 GB
    print("Computing branch statistics (mmap'd)...")
    branch_stats = np.zeros((199, 7), dtype=np.float32)
    for i in range(199):
        f0 = Xb[i, 0, :, :, cyt_idx].astype(np.float32)  # first frame
        branch_stats[i, 0] = float(np.max(f0))
        branch_stats[i, 1] = float(np.mean(f0))
        branch_stats[i, 2] = float(np.std(f0))
        ys, xs = np.mgrid[0:GRID, 0:GRID]
        total   = float(np.sum(np.abs(f0))) + 1e-12
        branch_stats[i, 3] = float(np.sum(xs * np.abs(f0))) / total  # centroid x
        branch_stats[i, 4] = float(np.sum(ys * np.abs(f0))) / total  # centroid y
        branch_stats[i, 5] = float(np.sum(f0 > 0.01)) / G2           # extent
        branch_stats[i, 6] = t_norm[i]

    return branch_stats, Xt, Yt, Yraw, masks, clip


def build_model(p, hidden):
    # Branch net
    b_in  = tf.keras.Input(shape=(7,), name="branch_in")
    b     = tf.keras.layers.Dense(hidden, activation="relu")(b_in)
    b_out = tf.keras.layers.Dense(p, activation="linear")(b)

    # Trunk net (gated)
    t_in  = tf.keras.Input(shape=(None, 24), name="trunk_in")
    U     = tf.keras.layers.Dense(hidden, activation="tanh")(t_in)
    V     = tf.keras.layers.Dense(hidden, activation="tanh")(t_in)
    h     = tf.keras.layers.Dense(hidden, activation="relu")(t_in)
    h     = tf.keras.layers.Dense(hidden, activation="linear")(h)
    h     = h * U + (1.0 - h) * V
    h     = tf.keras.layers.Dense(hidden, activation="relu")(h)
    h     = tf.keras.layers.Dense(hidden, activation="linear")(h)
    h     = h * U + (1.0 - h) * V
    t_out = tf.keras.layers.Dense(p, activation="linear")(h)   # (B,G²,p)

    bias  = tf.Variable(tf.zeros([1]), trainable=True, name="dot_bias")

    class DotLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            bv, tv = inputs
            return tf.einsum("bp,bnp->bn", bv, tv) + bias

    out = DotLayer()([b_out, t_out])                           # (B, G²)
    out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(out)  # (B, G², 1)
    return tf.keras.Model(inputs=[b_in, t_in], outputs=out)


def chunked_predict(model, b_feat, trunk_full, chunk=CHUNK_SIZE):
    """Predict in spatial chunks to avoid VRAM OOM."""
    results = []
    for start in range(0, G2, chunk):
        end   = min(start + chunk, G2)
        t_ch  = trunk_full[:, start:end, :]            # (B, chunk, 24)
        pred  = model([b_feat, t_ch], training=False)  # (B, chunk, 1)
        results.append(pred.numpy())
    return np.concatenate(results, axis=1)             # (B, G², 1)


def build_trunk_input(Xb_mmap, Xt, indices):
    """Build (len(indices), G², 24) trunk input from mmap'd X_branch."""
    B  = len(indices)
    xy = Xt[indices]                                    # (B, G², 3) — x,y,t
    # Spatial features: X_branch frames → (B, 2, G, G, 11) → (B, G², 22)
    sf = Xb_mmap[indices].astype(np.float32)           # (B, 2, G, G, 11)
    sf = sf.reshape(B, G2, 22)
    # Use only (x,y) from trunk; discard t (it's in branch already)
    return np.concatenate([xy[:, :, :2], sf], axis=-1) # (B, G², 24)


def train_model(model, Xb_mmap, Xt, Yt, bs, lr, Xb_stats):
    opt = tf.keras.optimizers.Adam(lr)
    tr_idx = np.arange(140)
    vl_idx = np.arange(140, 160)
    best_val, stagnant, best_w = 1e9, 0, None

    for epoch in range(FULL_EPOCHS):
        np.random.shuffle(tr_idx)
        for start in range(0, 140, bs):
            idx   = tr_idx[start:start+bs]
            t_inp = build_trunk_input(Xb_mmap, Xt, idx).astype(np.float32)
            y_inp = Yt[idx].reshape(len(idx), G2, 1).astype(np.float32)
            b_inp = Xb_stats[idx]
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((model([b_inp, t_inp]) - y_inp)**2)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        # Validation
        t_val   = build_trunk_input(Xb_mmap, Xt, vl_idx).astype(np.float32)
        y_val   = Yt[vl_idx].reshape(20, G2, 1).astype(np.float32)
        vl_pred = chunked_predict(model, Xb_stats[vl_idx], t_val)
        vl      = float(np.mean((vl_pred - y_val) ** 2))

        if vl < best_val - 1e-6:
            best_val, stagnant, best_w = vl, 0, model.get_weights()
        else:
            stagnant += 1
            if stagnant % LR_PATIENCE == 0:
                opt.learning_rate.assign(opt.learning_rate * 0.5)
            if stagnant >= PATIENCE:
                break

    if best_w:
        model.set_weights(best_w)
    return model


def compute_metrics(y_true, y_pred, masks, clip_max):
    yt = np.clip(y_true, 0, None)
    yp = np.clip(y_pred, 0, None)
    r2 = float(r2_score(yt.flatten(), yp.flatten()))
    rmse_u = float(np.sqrt(np.mean((yt - yp)**2)))
    m = masks.sum(-1) > 0  # any cell present
    if m.sum() > 0:
        rmse_m = float(np.sqrt(np.sum((yt - yp)**2 * m) / (m.sum() + 1e-12)))
    else:
        rmse_m = rmse_u
    ssim_vals = []
    for t in range(yt.shape[0]):
        if yt[t].std() > 1e-12:
            ssim_vals.append(float(ssim(yt[t], yp[t], data_range=clip_max)))
    ssim_avg = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    return {"Global_R2": r2, "Unmasked_RMSE": rmse_u, "Masked_RMSE": rmse_m, "SSIM": ssim_avg}


# ── main ───────────────────────────────────────────────────────────────────
def run(cyt_name, seed):
    print(f"\n{'='*60}")
    print(f"DeepONet 500x500 | cytokine={cyt_name} | seed={seed}")
    print(f"{'='*60}")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cyt_idx = CYT_MAP[cyt_name]
    Xb_stats, Xt, Yt, Yraw, masks, clip = load_data(cyt_idx)
    Xb_mmap = np.load(f"{DATA_DIR}/X_branch.npy", mmap_mode="r")

    def objective(trial):
        p   = trial.suggest_categorical("p",    [64, 128])
        h   = trial.suggest_categorical("hidden", [128, 256])
        lr  = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        bs  = trial.suggest_categorical("batch_size", [4, 8])
        m   = build_model(p, h)
        opt = tf.keras.optimizers.Adam(lr)
        tr_idx = np.arange(140)
        for ep in range(TUNE_EPOCHS):
            np.random.shuffle(tr_idx)
            for s in range(0, 140, bs):
                idx   = tr_idx[s:s+bs]
                t_inp = build_trunk_input(Xb_mmap, Xt, idx).astype(np.float32)
                y_inp = Yt[idx].reshape(len(idx), G2, 1).astype(np.float32)
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean((m([Xb_stats[idx], t_inp]) - y_inp)**2)
                opt.apply_gradients(zip(tape.gradient(loss, m.trainable_variables), m.trainable_variables))
        vl_idx   = np.arange(140, 160)
        t_val    = build_trunk_input(Xb_mmap, Xt, vl_idx).astype(np.float32)
        y_val    = Yt[vl_idx].reshape(20, G2, 1).astype(np.float32)
        vl_pred  = chunked_predict(m, Xb_stats[vl_idx], t_val)
        val_loss = float(np.mean((vl_pred - y_val) ** 2))
        del m; gc.collect()
        return float(val_loss)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    best = study.best_params
    print(f"Best params: {best}")

    model = build_model(best["p"], best["hidden"])
    t_train_start = time.time()
    model = train_model(model, Xb_mmap, Xt, Yt, best["batch_size"], best["lr"], Xb_stats)
    train_elapsed = time.time() - t_train_start

    # Evaluate on test set
    test_idx  = np.arange(160, 199)
    t_test    = build_trunk_input(Xb_mmap, Xt, test_idx).astype(np.float32)
    t_pred_start = time.time()
    pred_flat = chunked_predict(model, Xb_stats[test_idx], t_test)  # (39,G²,1)
    pred_elapsed = time.time() - t_pred_start
    pred      = pred_flat.reshape(39, GRID, GRID)
    y_true    = Yraw[160:199]
    pred_phys = (pred + 1.0) / 2.0 * clip
    pred_phys = np.clip(pred_phys, 0, None)

    metrics = compute_metrics(y_true, pred_phys, masks[160:199], clip)
    metrics.update({"cytokine": cyt_name, "seed": seed, "grid": 500,
                    "train_time_seconds": round(train_elapsed, 2),
                    "pred_time_seconds":  round(pred_elapsed, 4),
                    "best_params": best, "model": "deeponet_h"})

    out_path = f"{RESULTS_DIR}/res_{cyt_name}_500_{seed}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {out_path}")
    print(f"Results: {metrics}")

    del model, Xb_mmap; gc.collect()


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
