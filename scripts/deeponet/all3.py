import os, json, argparse, random, time
from pathlib import Path

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------
# GPU setup (A100: enable memory_growth + bf16 mixed precision)
# ----------------------------------------------------------------------
def configure_gpu(use_mixed_precision=True):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"  [warn] couldn't set memory_growth on {gpu}: {e}")
    if gpus:
        print(f"  [gpu] {len(gpus)} GPU(s) available")
        if use_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
                print(f"  [gpu] mixed precision: mixed_bfloat16 (A100 optimised)")
            except Exception:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        print(f"  [gpu] NO GPU DETECTED — running on CPU (slow).")


configure_gpu()


N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400
EVAL_CHUNK  = 4096


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

# branch
class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p,      activation="linear")

    def call(self, x, training=False):
        return self.fc2(self.fc1(x))


# trunk
class Trunk(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.U   = tf.keras.layers.Dense(hidden, activation="tanh")
        self.V   = tf.keras.layers.Dense(hidden, activation="tanh")
        self.W1a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W1b = tf.keras.layers.Dense(hidden, activation="linear")
        self.W2a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W2b = tf.keras.layers.Dense(hidden, activation="linear")
        self.out = tf.keras.layers.Dense(p,      activation="linear")

    def call(self, x):
        u = self.U(x); v = self.V(x)
        h = self.W1b(self.W1a(x));  h = h * u + (1.0 - h) * v
        h = self.W2b(self.W2a(h));  h = h * u + (1.0 - h) * v
        return self.out(h)

class DeepONet(tf.keras.Model):
    def __init__(self, hidden, p):
        super().__init__()
        self.branch = Branch(hidden, p)
        self.trunk  = Trunk(hidden, p)
        self.bias   = self.add_weight(shape=(1,), initializer="zeros",
                                      trainable=True, name="bias")

    def call(self, inputs, training=False):
        xb, xt = inputs
        b = self.branch(xb, training=training)         # (batch, p)
        t = self.trunk(xt)                             # (batch, n_pts, p)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias  # (batch, n_pts)
        # cast back to float32 for numerically stable loss under mixed precision
        return tf.cast(tf.expand_dims(r, -1), tf.float32)   # (batch, n_pts, 1)


def build_branch_inputs(Xb, Xt, cyt_idx):
    """
    Branch input: 8 scalar features per sample summarising frame 0
    of the chosen cytokine + its 3D spatial extent + time.
        0: max_f0     ∈ [0,1]   (rescaled from [-1,1])
        1: mean_f0    ∈ [0,1]
        2: std_f0     ∈ [0,1]
        3: centroid_x ∈ [0,1]
        4: centroid_y ∈ [0,1]
        5: centroid_z ∈ [0,1]
        6: extent     ∈ [0,1]   (fraction of voxels with cells present)
        7: t_norm     ∈ [-1,1]
    """
    N, _, G, _, _, _ = Xb.shape
    f0   = Xb[:, 0, :, :, :, cyt_idx]                                 # (N, G, G, G)
    mask = (Xb[:, 0, :, :, :, 6:].max(axis=-1) > 0.5).astype(np.float32)  # (N, G, G, G)

    xs = np.linspace(0, 1, G, dtype=np.float32)
    ys = np.linspace(0, 1, G, dtype=np.float32)
    zs = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')

    out = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        f  = f0[i]; m = mask[i]; na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(f))  + 1.0) / 2.0
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0
        out[i, 2] = float(np.std(f))
        out[i, 3] = float(np.sum(xx * m) / na)
        out[i, 4] = float(np.sum(yy * m) / na)
        out[i, 5] = float(np.sum(zz * m) / na)
        out[i, 6] = na / (G * G * G)
        out[i, 7] = float(Xt[i, 0, 3])  # t_norm (same for all pts)
    return out


def build_trunk_inputs(Xb, Xt):
    """
    Trunk input per point: (x, y, z) + 22 stacked channel values from the
    2-frame window  →  25-D feature vector.
    """
    N, _, G, _, _, C = Xb.shape
    # Xb: (N, 2, G, G, G, 11) → permute frames+channels last, then flatten spatial
    vals = Xb.transpose(0, 2, 3, 4, 1, 5).reshape(N, G*G*G, 22).astype(np.float32)
    xyz  = Xt[:, :, :3].astype(np.float32)                            # (N, G³, 3)
    return np.concatenate([xyz, vals], axis=-1)                        # (N, G³, 25)


def build_dataset(Xbranch, Xtrunk, Yf, batch_size, chunk_size, shuffle=True):
    N, n_pts, _ = Xtrunk.shape
    chunks = list(range(0, n_pts, chunk_size))

    def gen():
        order = np.arange(N)
        if shuffle:
            np.random.shuffle(order)
        for i in order:
            xb = Xbranch[i]
            for s in chunks:
                e    = min(s + chunk_size, n_pts)
                size = e - s
                xt = Xtrunk[i, s:e]
                y  = Yf[i, s:e]
                if size < chunk_size:
                    pad = chunk_size - size
                    xt = np.concatenate([xt, np.zeros((pad, 25), np.float32)], axis=0)
                    y  = np.concatenate([y,  np.zeros((pad, 1),  np.float32)], axis=0)
                sz = np.array([size], dtype=np.int32)
                yield (xb, xt, sz), y

    sig = (
        (tf.TensorSpec((8,),              tf.float32),
         tf.TensorSpec((chunk_size, 25),  tf.float32),
         tf.TensorSpec((1,),              tf.int32)),
        tf.TensorSpec((chunk_size, 1),    tf.float32),
    )
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))

def masked_mse(pred, y, sz):
    idx  = tf.range(tf.shape(pred)[1])[tf.newaxis, :, tf.newaxis]
    mask = tf.cast(idx < tf.cast(sz[:, tf.newaxis, :], tf.int32), tf.float32)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)


def _make_steps(model, opt):
    """
    Build train/val tf.functions BOUND to a specific model + optimizer pair.

    This is the fix for the XLA retracing hang: when the script runs multiple
    Optuna trials (or sequential runs of different cytokines), each trial
    creates a fresh model. A module-level @tf.function would try to retrace
    on every new model, hitting XLA's "tf.Variable created inside tf.function"
    error. Building tf.functions per-model avoids this entirely.
    """
    @tf.function(reduce_retracing=True)
    def train_step(xb, xt, sz, y):
        with tf.GradientTape() as tape:
            loss = masked_mse(model([xb, xt], training=True), y, sz)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def val_step(xb, xt, sz, y):
        return masked_mse(model([xb, xt], training=False), y, sz)

    return train_step, val_step


# Backwards-compat eager versions (used by tests that call them directly).
def train_step(model, opt, xb, xt, sz, y):
    with tf.GradientTape() as tape:
        loss = masked_mse(model([xb, xt], training=True), y, sz)
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))
    return float(loss)


def val_step(model, xb, xt, sz, y):
    return float(masked_mse(model([xb, xt], training=False), y, sz))


def train_model(model, opt, ds_tr, ds_vl,
                epochs, patience=40, reduce_patience=15, min_lr=1e-7,
                verbose=True):
    # Eager warmup — build all tf.Variables BEFORE the first tf.function call,
    # otherwise variable creation inside the traced graph is forbidden.
    for (xb, xt, sz), y in ds_tr.take(1):
        _ = model([xb, xt], training=False)
        break

    train_step_fn, val_step_fn = _make_steps(model, opt)

    best_val = np.inf; best_w = None; wait = rw = 0
    for ep in range(1, epochs + 1):
        tr_losses = [float(train_step_fn(xb, xt, sz, y))
                     for (xb, xt, sz), y in ds_tr]
        vl_losses = [float(val_step_fn(xb, xt, sz, y))
                     for (xb, xt, sz), y in ds_vl]
        tr = float(np.mean(tr_losses)); vl = float(np.mean(vl_losses))
        if verbose and ep % 20 == 0:
            print(f"  Epoch {ep:4d}  loss={tr:.5f}  val={vl:.5f}")
        if vl < best_val:
            best_val = vl; best_w = model.get_weights(); wait = rw = 0
        else:
            wait += 1; rw += 1
        if rw >= reduce_patience:
            lr = float(opt.learning_rate)
            new_lr = max(lr * 0.5, min_lr)
            if new_lr != lr:
                opt.learning_rate.assign(new_lr)
                if verbose: print(f"  LR → {new_lr:.2e}")
            rw = 0
        if wait >= patience:
            if verbose: print(f"  Early stop @ epoch {ep}")
            break
    if best_w:
        model.set_weights(best_w)
    return best_val


# eval
def predict_full(model, Xbranch, Xtrunk, chunk=EVAL_CHUNK):
    N, n_pts, _ = Xtrunk.shape
    out = np.zeros((N, n_pts, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i+1])                    # (1, 8)
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            xt = tf.constant(Xtrunk[i:i+1, s:e])            # (1, e-s, 25)
            out[i, s:e] = model([xb, xt], training=False).numpy()[0]
    return out


# metrics

def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def _inv_fisher_z(z):
    return float(np.tanh(z))


def compute_2d_slice_metrics(yt, yp, clip_max):
    """
    Per-axis 2D mid-slice metrics.

    Addresses the supervisor's first question — "in 3D, only 2D solutions
    matter, biologists only see 2D slices". For each axis (xy mid-plane,
    xz mid-plane, yz mid-plane) we compute R² and SSIM averaged across the
    T time steps. This tells us how good the model is *as a 2D-slice
    predictor*, which is the actually-observable quantity.

    yt, yp : (T, G, G, G, 1)
    """
    T = yt.shape[0]; G = yt.shape[1]
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    mid = G // 2

    out = {}
    for axis_name, sl in (("xy_midplane_z",  np.s_[:, :, :, mid, 0]),
                          ("xz_midplane_y",  np.s_[:, :, mid, :, 0]),
                          ("yz_midplane_x",  np.s_[:, mid, :, :, 0])):
        gts = yt[sl]   # (T, G, G)
        prs = yp[sl]
        r2s, ssims, n_skip = [], [], 0
        for t in range(T):
            gt = gts[t]; pr = prs[t]
            if np.std(gt) > 1e-12:
                r2s.append(float(r2_score(gt.flatten(), pr.flatten())))
            else:
                n_skip += 1
            dr = float(np.max(gt) - np.min(gt))
            if dr > 1e-12:
                ssims.append(float(ssim(gt, pr, data_range=fixed_dr)))
        out[axis_name] = {
            "R2":   float(np.mean(r2s))   if r2s   else 0.0,
            "SSIM": float(np.mean(ssims)) if ssims else 0.0,
            "Skipped_Frames": n_skip,
        }
    return out


def calculate_metrics(y_true, y_pred, masks, clip_max):
    """
    y_true, y_pred: (T, G, G, G, 1)
    masks         : (T, G, G, G, 5)  — cell-type one-hot masks per voxel
    """
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]; yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)  # collapse 5 cell-type channels → 1

    sq_diff = np.square(yt - yp)
    rmse = float(np.sqrt(np.sum(sq_diff * ms) / (np.sum(ms) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))
    r2   = float(r2_score(yt.flatten(), yp.flatten()))

    per_t_r2 = []
    for t in range(T):
        gt_f = yt[t].flatten(); pr_f = yp[t].flatten()
        per_t_r2.append(float(r2_score(gt_f, pr_f)) if np.std(gt_f) > 1e-12 else np.nan)

    dice_thr = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices = []; n_empty = 0
    z_corrs = []
    ssims_v = []; n_ssim_skip = 0
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0

    for t in range(T):
        gt = yt[t, :, :, :, 0]; pr = yp[t, :, :, :, 0]   # (G, G, G)
        gb = (gt > dice_thr).astype(float); pb = (pr > dice_thr).astype(float)
        if np.sum(gb) + np.sum(pb) == 0:
            n_empty += 1
        else:
            dices.append((2.0 * np.sum(gb * pb)) / (np.sum(gb) + np.sum(pb) + 1e-12))
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))
        dr = float(np.max(gt) - np.min(gt))
        if dr > 1e-12:
            # skimage SSIM works on N-D arrays; explicit data_range for stability
            ssims_v.append(float(ssim(gt, pr, data_range=fixed_dr)))
        else:
            n_ssim_skip += 1

    return {
        "Global_R2":           r2,
        "Per_Timestep_R2":     per_t_r2,
        "Masked_RMSE":         rmse,
        "Unmasked_RMSE":       unmasked_rmse,
        "Avg_Dice":            float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped":  n_empty,
        "Spatial_Correlation": _inv_fisher_z(float(np.mean(z_corrs))) if z_corrs else 0.0,
        "SSIM":                float(np.mean(ssims_v)) if ssims_v else 0.0,
        "SSIM_Skipped_Frames": n_ssim_skip,
        "Slice_2D":            compute_2d_slice_metrics(yt, yp, clip_max),
    }

def denormalize(x, clip_max):
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * clip_max

# optuna
def make_objective(Xbr_tr, Xtr_tr, Yf_tr,
                   Xbr_vl, Xtr_vl, Yf_vl, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        p          = trial.suggest_categorical("p",          [64, 128, 256])
        hidden     = trial.suggest_categorical("hidden",     [128, 256])
        lr         = trial.suggest_float("learning_rate",    1e-5, 1e-3, log=True)
        bs         = trial.suggest_categorical("batch_size", [2, 4])
        chunk_size = trial.suggest_categorical("chunk_size", [4096, 8192])

        ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr, bs, chunk_size, shuffle=True)
        ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl, bs, chunk_size, shuffle=False)

        model = DeepONet(hidden=hidden, p=p)
        opt   = tf.keras.optimizers.Adam(lr)
        best  = train_model(model, opt, ds_tr, ds_vl,
                            epochs=TUNE_EPOCHS, patience=8,
                            reduce_patience=5, verbose=False)
        return float(best)
    return objective

def run_pipeline(grid, seed, cytokine, data_root="./preprocessed_3d",
                 out_root="./models/deeponet_h_3d"):
    set_seed(seed)
    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx = cyt_names.index(cytokine.lower())

    data_path = Path(f"{data_root}/{grid}x{grid}x{grid}")
    out_dir   = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid}x{grid} — loading data...")
    Xb = np.load(data_path/"X_branch.npy").astype(np.float32)        # (N, 2, G, G, G, 11)
    Xt = np.load(data_path/"X_trunk.npy").astype(np.float32)         # (N, G³, 4)
    Y  = np.load(data_path/"Y_target.npy").astype(np.float32)[..., idx:idx+1]  # (N,G,G,G,1)
    M  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float32) # (N, G, G, G, 5)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    N = Xb.shape[0]; G3 = Xt.shape[1]; G = int(round(G3 ** (1.0/3.0)))
    Yf = Y.reshape(N, G3, 1)

    Xbranch = build_branch_inputs(Xb, Xt, idx)
    Xtrunk  = build_trunk_inputs(Xb, Xt)

    print(f"  Branch input: (N, 8) scalars  |  Trunk input: (N, {G3}, 25)")
    print(f"  Full grid per epoch: {G3} pts × {N} samples - no subsampling")

    # 70/10/20
    Xbr_tr, Xtr_tr, Yf_tr = Xbranch[:70],   Xtrunk[:70],   Yf[:70]
    Xbr_vl, Xtr_vl, Yf_vl = Xbranch[70:80], Xtrunk[70:80], Yf[70:80]

    if seed == 42:
        print(f"Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            make_objective(Xbr_tr, Xtr_tr, Yf_tr, Xbr_vl, Xtr_vl, Yf_vl, 42),
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

    tf.keras.backend.clear_session(); set_seed(seed)

    ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr,
                          best["batch_size"], best["chunk_size"], shuffle=True)
    ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl,
                          best["batch_size"], best["chunk_size"], shuffle=False)

    model = DeepONet(hidden=best["hidden"], p=best["p"])
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])
    print(f"Final training [{cytokine.upper()}] {grid}x{grid}x{grid}  (max {FULL_EPOCHS} epochs)...")

    # train
    t_train_start = time.time()
    train_model(model, opt, ds_tr, ds_vl,
                epochs=FULL_EPOCHS, patience=40, reduce_patience=15, verbose=True)
    train_elapsed = time.time() - t_train_start
    print(f"  Training time: {train_elapsed:.1f}s")

    # predict
    t_pred_start = time.time()
    Yp_flat = predict_full(model, Xbranch, Xtrunk)
    pred_elapsed = time.time() - t_pred_start
    print(f"  Prediction time (all {N} samples): {pred_elapsed:.1f}s")

    Yp      = Yp_flat.reshape(N, G, G, G, 1)
    Y_phys  = denormalize(Y.reshape(N, G, G, G, 1), clip_max)
    Yp_phys = denormalize(Yp, clip_max)

    # test windows
    suffix  = f"{cytokine}_{grid}_{seed}"

    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params":          best,
        "optuna_best_val_loss": optuna_val,
        "train_time_seconds":   round(train_elapsed, 2),
        "pred_time_seconds":    round(pred_elapsed,  2),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Y_phys[80:90], Yp_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Y_phys[90:99], Yp_phys[90:99], M[90:99], clip_max),
        },
    }
    with open(out_dir/f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir/f"weights_{suffix}.weights.h5")
    print(f"DONE → {out_dir}/res_{suffix}.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid",     type=int, default=None)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--data",     type=str, default="./preprocessed_3d",
                    help="Root folder containing <G>x<G>x<G>/*.npy. Use "
                         "./preprocessed_3d_grayscott for the PhysicsNemo benchmark.")
    ap.add_argument("--out",      type=str, default="./models/deeponet_h_3d",
                    help="Output folder for weights + JSON.")
    args = ap.parse_args()

    if args.grid:
        run_pipeline(args.grid, args.seed, args.cytokine,
                     data_root=args.data, out_root=args.out)
    else:
        for d in sorted(Path(args.data).iterdir()):
            if d.is_dir():
                run_pipeline(int(d.name.split("x")[0]), args.seed, args.cytokine,
                             data_root=args.data, out_root=args.out)
