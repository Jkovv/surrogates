"""
DeepONet++ (3D) — architectural extensions over baseline DeepONet for 3D
cytokine / reaction-diffusion surrogate modelling.

Four upgrades over `deeponet_3d.py`, each with a clear scientific justification:

  1. CNN branch encoder (replaces 8 hand-crafted scalars)
     The baseline fed the branch 8 summary stats (max/mean/std + centroid + t).
     That is an input bottleneck — the model never "sees" the spatial shape of
     the wound, only moments of it. Here a lightweight 3D CNN processes the
     full t=0 concentration volume + cell-type masks and produces a latent
     vector of size p. This lets the operator condition on geometry.

  2. Factorized Axial Attention (FAA) in the trunk
     The baseline trunk was a pointwise MLP — every voxel is processed in
     isolation. FAA processes x-axis, y-axis, z-axis separately via attention
     and fuses them. Cost: O(G^2) per axis instead of O(G^6) for full 3D
     attention. Scientific motivation: a biologist only observes 2D slices,
     so the *z-coupling* is the interesting learned quantity. FAA exposes it
     explicitly and makes it ablatable (you can zero out the z-head and
     measure the accuracy drop).

  3. Linear-Nonlinear Fusion (LNF) head
     The field u(x,t) obeys a PDE with two distinct parts: a linear diffusion
     operator (smooth, long-range) and a nonlinear reaction/source term
     (sharp, local — cells secrete cytokines at specific voxels). LNF
     separates these: one trunk branch is purely linear, one is nonlinear
     with ReLU/GELU activations, and they combine by elementwise product.
     Empirically this gives better accuracy at lower FLOPs (see LNF-NO 2025).

  4. Causal time-weighted loss
     Long rollouts in reaction-diffusion suffer from error accumulation.
     We upweight early timesteps in the loss, so the model masters the
     initial dynamics before the late ones (PDE-Refiner style).

The output shape and JSON metrics are identical to `deeponet_3d.py` so your
existing VTK export and evaluation plumbing works unchanged.
"""
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
# GPU / mixed precision setup
# ----------------------------------------------------------------------
def configure_gpu(use_mixed_precision=True):
    """Enable memory growth, TF32, and bf16 mixed precision on A100."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"  [warn] couldn't set memory_growth on {gpu}: {e}")
    if gpus:
        print(f"  [gpu] {len(gpus)} GPU(s) available: {[g.name for g in gpus]}")
        if use_mixed_precision:
            # bf16 is preferred on A100 (Ampere) — no loss scaling needed.
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
                print(f"  [gpu] mixed precision: mixed_bfloat16 (A100 optimised)")
            except Exception:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print(f"  [gpu] mixed precision: mixed_float16 (fallback)")
    else:
        print(f"  [gpu] NO GPU DETECTED — running on CPU (slow).")
    return bool(gpus)


configure_gpu()

N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400
EVAL_CHUNK  = 4096


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


# ----------------------------------------------------------------------
# Branch: 3D CNN encoder
# ----------------------------------------------------------------------
class CNNBranch(tf.keras.layers.Layer):
    """
    Encodes the full t=0 multi-channel volume into a latent of size p.

    Input : (batch, G, G, G, 11)    (6 cytokines + 5 cell-type masks at t=0)
    Output: (batch, p)
    """
    def __init__(self, p, base_filters=16, **kw):
        super().__init__(**kw)
        self.c1 = tf.keras.layers.Conv3D(base_filters,   3, padding="same", activation="relu")
        self.c2 = tf.keras.layers.Conv3D(base_filters*2, 3, strides=2, padding="same", activation="relu")
        self.c3 = tf.keras.layers.Conv3D(base_filters*4, 3, strides=2, padding="same", activation="relu")
        self.gap = tf.keras.layers.GlobalAveragePooling3D()
        self.fc  = tf.keras.layers.Dense(p, activation="linear")

    def call(self, x, training=False):
        x = self.c1(x); x = self.c2(x); x = self.c3(x)
        x = self.gap(x)
        return self.fc(x)


# ----------------------------------------------------------------------
# Factorized Axial Attention block
# ----------------------------------------------------------------------
class AxialAttention(tf.keras.layers.Layer):
    """
    Attention along a single axis of a 3D volume.

    Input  : (batch, G, G, G, C)
    axis ∈ {1, 2, 3}  — which spatial axis to attend over
    Output : same shape as input
    """
    def __init__(self, dim, num_heads=4, axis=1, **kw):
        super().__init__(**kw)
        self.axis = axis
        self.mha  = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        # x shape: (B, G, G, G, C)
        # move target axis to position 1, flatten the other two spatial axes
        # into the batch dimension, run attention along the axis, reshape back.
        shape = tf.shape(x); B = shape[0]; Gx = shape[1]; Gy = shape[2]; Gz = shape[3]; C = x.shape[-1]
        if self.axis == 1:   # attend over x
            perm = [0, 2, 3, 1, 4]; inv = [0, 3, 1, 2, 4]; L = Gx; other = (Gy, Gz)
        elif self.axis == 2: # attend over y
            perm = [0, 1, 3, 2, 4]; inv = [0, 1, 3, 2, 4]; L = Gy; other = (Gx, Gz)
        else:                # attend over z
            perm = [0, 1, 2, 3, 4]; inv = [0, 1, 2, 3, 4]; L = Gz; other = (Gx, Gy)

        x_p   = tf.transpose(x, perm)                                        # (B, o1, o2, L, C)
        x_r   = tf.reshape(x_p, (B * other[0] * other[1], L, C))             # (B·o1·o2, L, C)
        attn  = self.mha(x_r, x_r)                                           # same shape
        x_r   = self.norm(x_r + attn)                                        # residual + LN
        x_p2  = tf.reshape(x_r, (B, other[0], other[1], L, C))
        return tf.transpose(x_p2, inv)                                       # back to (B, G, G, G, C)


class FAATrunk(tf.keras.layers.Layer):
    """
    Factorized axial attention trunk. Instead of a pointwise MLP, apply
    attention along x, y, z in sequence so that slice coupling is explicit.

    Input : per-voxel features laid out as a full volume (B, G, G, G, F_in)
    Output: (B, G, G, G, p)
    """
    def __init__(self, dim, p, num_heads=4, **kw):
        super().__init__(**kw)
        self.project_in  = tf.keras.layers.Dense(dim, activation="gelu")
        self.attn_x = AxialAttention(dim, num_heads=num_heads, axis=1)
        self.attn_y = AxialAttention(dim, num_heads=num_heads, axis=2)
        self.attn_z = AxialAttention(dim, num_heads=num_heads, axis=3)
        self.mlp    = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation="gelu"),
            tf.keras.layers.Dense(p),
        ])

    def call(self, x):
        h = self.project_in(x)
        h = self.attn_x(h)
        h = self.attn_y(h)
        h = self.attn_z(h)
        return self.mlp(h)


# ----------------------------------------------------------------------
# DeepONet++ model with LNF fusion
# ----------------------------------------------------------------------
class DeepONetPP(tf.keras.Model):
    """
    Output u(x,y,z) = bias + ( Branch(x_br) ⊙ Trunk_L(x_tr) ) ⊙ ReLU(Trunk_N(x_tr))

    where:
      - Branch   : 3D CNN encoder over t=0 volume              → (B, p)
      - Trunk_L  : factorized axial attention (linear head)    → (B, G, G, G, p)
      - Trunk_N  : factorized axial attention (nonlinear head) → (B, G, G, G, p)
      - ⊙        : elementwise product across the p dimension
    """
    def __init__(self, hidden, p, num_heads=4):
        super().__init__()
        self.branch  = CNNBranch(p=p, base_filters=16)
        self.trunk_L = FAATrunk(dim=hidden, p=p, num_heads=num_heads)
        self.trunk_N = FAATrunk(dim=hidden, p=p, num_heads=num_heads)
        self.bias    = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="bias")
        self.p       = p

    def call(self, inputs, training=False):
        xb, xt = inputs                                              # xb: (B, G, G, G, 11), xt: (B, G, G, G, F_tr)
        b     = self.branch(xb, training=training)                   # (B, p)
        tL    = self.trunk_L(xt)                                     # (B, G, G, G, p)
        tN    = tf.nn.relu(self.trunk_N(xt))                         # (B, G, G, G, p)
        # broadcast branch to spatial dims and fuse
        b_exp = b[:, None, None, None, :]                            # (B, 1, 1, 1, p)
        linear_part    = tf.reduce_sum(b_exp * tL, axis=-1, keepdims=True)  # (B, G, G, G, 1)
        nonlinear_part = tf.reduce_sum(b_exp * tN, axis=-1, keepdims=True)  # (B, G, G, G, 1)
        out = linear_part * nonlinear_part + self.bias               # (B, G, G, G, 1)
        # Under mixed precision the computation happens in bf16; cast back to
        # float32 at the very end so loss + metrics stay numerically stable.
        return tf.cast(out, tf.float32)


# ----------------------------------------------------------------------
# Input builders (shape-compatible with preprocess_3d output)
# ----------------------------------------------------------------------
def build_branch_inputs_volumetric(Xb, cyt_idx):
    """
    Volumetric branch input: the t=0 frame of all 11 channels (6 cytokines + 5 masks).
    Shape: (N, G, G, G, 11)

    cyt_idx is accepted for API symmetry with the baseline but is NOT used to
    subset channels — the CNN branch gets all 11 channels and learns which are
    relevant to the target cytokine.
    """
    return Xb[:, 0].astype(np.float32)                               # (N, G, G, G, 11)


def build_trunk_inputs_volumetric(Xb, Xt):
    """
    Volumetric trunk input: per-voxel stack of (x, y, z, 22 channels from the 2-frame window).

    Shape: (N, G, G, G, 25)
    """
    N, _, G, _, _, _ = Xb.shape
    # reshape trunk coords back to volume
    xyz_flat = Xt[:, :, :3].astype(np.float32)                       # (N, G³, 3)
    xyz_vol  = xyz_flat.reshape(N, G, G, G, 3)
    # stack 2 frames × 11 channels → 22 along last axis at each voxel
    vals     = Xb.transpose(0, 2, 3, 4, 1, 5).reshape(N, G, G, G, 22).astype(np.float32)
    return np.concatenate([xyz_vol, vals], axis=-1)                  # (N, G, G, G, 25)


# ----------------------------------------------------------------------
# Dataset — GPU-preloaded tensors (faster than generator for small N)
# ----------------------------------------------------------------------
def build_dataset(Xbranch, Xtrunk, Y_vol, t_norm, batch_size, shuffle=True):
    """
    For ~70 samples the whole dataset fits easily in GPU RAM:
      70 * 50^3 * 25 float32  ≈ 0.9 GB
      70 * 50^3 * 11 float32  ≈ 0.4 GB
    We preload to GPU to avoid H2D copies each batch, then use tf.data for
    shuffle + batch. t_norm is bundled into the tuple so training step
    doesn't need external bookkeeping.
    """
    ds = tf.data.Dataset.from_tensor_slices(
        ((Xbranch.astype(np.float32), Xtrunk.astype(np.float32),
          t_norm.astype(np.float32)),
         Y_vol.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(len(Xbranch), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)




def _causal_se(pred, y, t_norm, lam):
    """Causal time-weighted squared error (mean reduction)."""
    w  = tf.exp(-lam * (t_norm + 1.0))                                # (B,)
    se = tf.square(pred - y)                                          # (B, G, G, G, 1)
    return tf.reduce_mean(w[:, None, None, None, None] * se)


def _make_steps(model, opt):
    """
    Build train/val tf.functions bound to a SPECIFIC model instance. Returns
    (train_step, val_step). Each Optuna trial creates a fresh model + fresh
    opt + fresh pair of tf.functions, so we never re-trace across instances
    (which is what was crashing trials 1+ before).
    """
    @tf.function(reduce_retracing=True)
    def train_step(xb, xt, y, t_norm, lam):
        with tf.GradientTape() as tape:
            pred = model([xb, xt], training=True)
            loss = _causal_se(pred, y, t_norm, lam)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def val_step(xb, xt, y, t_norm, lam):
        pred = model([xb, xt], training=False)
        return _causal_se(pred, y, t_norm, lam)

    return train_step, val_step


# Backwards-compat names so external scripts (e.g. unit tests) still work.
def causal_mse(pred, y, t_norm, lam):
    return _causal_se(pred, y, t_norm, lam)


def train_step(model, opt, xb, xt, y, t_norm, lam):
    """Eager fallback (used by tests). Production path uses _make_steps."""
    with tf.GradientTape() as tape:
        pred = model([xb, xt], training=True)
        loss = _causal_se(pred, y, t_norm, lam)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss)


def val_step(model, xb, xt, y, t_norm, lam):
    """Eager fallback (used by tests)."""
    pred = model([xb, xt], training=False)
    return float(_causal_se(pred, y, t_norm, lam))


# ----------------------------------------------------------------------
# Training loop (with early stop + reduce-on-plateau — same pattern as baseline)
# ----------------------------------------------------------------------
def train_model(model, opt, ds_tr, ds_vl,
                epochs, lam=0.5, patience=40, reduce_patience=15, min_lr=1e-7, verbose=True):
    # IMPORTANT: build all model variables eagerly with one warmup forward
    # pass BEFORE creating the tf.function. Otherwise the first traced call
    # creates new tf.Variables inside tf.function, which is forbidden and
    # was causing every Optuna trial after #0 to fail.
    for (xb, xt, _t), _y in ds_tr.take(1):
        _ = model([xb, xt], training=False)
        break

    train_step_fn, val_step_fn = _make_steps(model, opt)

    best_val = np.inf; best_w = None; wait = rw = 0
    for ep in range(1, epochs + 1):
        tr_losses = []
        for (xb, xt, t_b), y in ds_tr:
            tr_losses.append(float(train_step_fn(xb, xt, y, t_b, lam)))
        vl_losses = []
        for (xb, xt, t_b), y in ds_vl:
            vl_losses.append(float(val_step_fn(xb, xt, y, t_b, lam)))
        tr = float(np.mean(tr_losses)); vl = float(np.mean(vl_losses))
        if verbose and ep % 10 == 0:
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


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def predict_full(model, Xbranch, Xtrunk, batch=1):
    N = Xbranch.shape[0]; G = Xbranch.shape[1]
    out = np.zeros((N, G, G, G, 1), np.float32)
    for i in range(0, N, batch):
        j = min(i + batch, N)
        xb = tf.constant(Xbranch[i:j])
        xt = tf.constant(Xtrunk[i:j])
        out[i:j] = model([xb, xt], training=False).numpy()
    return out


# ----------------------------------------------------------------------
# Metrics (identical to baseline for direct comparison)
# ----------------------------------------------------------------------
def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def _inv_fisher_z(z):
    return float(np.tanh(z))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]; yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)

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
        gt = yt[t, :, :, :, 0]; pr = yp[t, :, :, :, 0]
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
    }


def denormalize(x, clip_max):
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * clip_max


# ----------------------------------------------------------------------
# Optuna
# ----------------------------------------------------------------------
def make_objective(Xbr_tr, Xtr_tr, Y_tr, t_tr,
                   Xbr_vl, Xtr_vl, Y_vl, t_vl, seed):
    def objective(trial):
        set_seed(seed); tf.keras.backend.clear_session()
        p         = trial.suggest_categorical("p",         [32, 64, 128])
        hidden    = trial.suggest_categorical("hidden",    [64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4])
        lr        = trial.suggest_float("learning_rate",   1e-5, 1e-3, log=True)
        bs        = trial.suggest_categorical("batch_size", [1, 2])
        lam       = trial.suggest_float("causal_lambda",   0.0, 1.0)

        ds_tr = build_dataset(Xbr_tr, Xtr_tr, Y_tr, t_tr, bs, shuffle=True)
        ds_vl = build_dataset(Xbr_vl, Xtr_vl, Y_vl, t_vl, bs, shuffle=False)

        model = DeepONetPP(hidden=hidden, p=p, num_heads=num_heads)
        opt   = tf.keras.optimizers.Adam(lr)
        best  = train_model(model, opt, ds_tr, ds_vl,
                            epochs=TUNE_EPOCHS, lam=lam,
                            patience=8, reduce_patience=5, verbose=False)
        return float(best)
    return objective


# ----------------------------------------------------------------------
# Pipeline — same data layout / split / output JSON schema as baseline
# ----------------------------------------------------------------------
def run_pipeline(grid, seed, cytokine, data_root="./preprocessed_3d",
                 out_root="./models/deeponet_pp_3d"):
    set_seed(seed)
    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx = cyt_names.index(cytokine.lower())

    data_path = Path(f"{data_root}/{grid}x{grid}x{grid}")
    out_dir   = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid}x{grid} — loading data...")
    Xb = np.load(data_path/"X_branch.npy").astype(np.float32)        # (N, 2, G, G, G, 11)
    Xt = np.load(data_path/"X_trunk.npy").astype(np.float32)         # (N, G³, 4)
    Y  = np.load(data_path/"Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float32)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    N = Xb.shape[0]; G = Xb.shape[2]

    # t_norm is shared across all voxels of a sample → take [:, 0, 3]
    t_all = Xt[:, 0, 3].astype(np.float32)                           # (N,)

    Xbranch = build_branch_inputs_volumetric(Xb, idx)                # (N, G, G, G, 11)
    Xtrunk  = build_trunk_inputs_volumetric(Xb, Xt)                  # (N, G, G, G, 25)

    print(f"  Branch  : (N, {G}, {G}, {G}, 11)   (CNN encoder)")
    print(f"  Trunk   : (N, {G}, {G}, {G}, 25)   (Factorized Axial Attention)")
    print(f"  Volumetric batching — no point subsampling")

    Xbr_tr, Xtr_tr, Y_tr, t_tr = Xbranch[:70],   Xtrunk[:70],   Y[:70],   t_all[:70]
    Xbr_vl, Xtr_vl, Y_vl, t_vl = Xbranch[70:80], Xtrunk[70:80], Y[70:80], t_all[70:80]

    if seed == 42:
        print(f"Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            make_objective(Xbr_tr, Xtr_tr, Y_tr, t_tr,
                           Xbr_vl, Xtr_vl, Y_vl, t_vl, 42),
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
        best = ref["best_params"]; optuna_val = ref["optuna_best_val_loss"]

    tf.keras.backend.clear_session(); set_seed(seed)

    ds_tr = build_dataset(Xbr_tr, Xtr_tr, Y_tr, t_tr, best["batch_size"], shuffle=True)
    ds_vl = build_dataset(Xbr_vl, Xtr_vl, Y_vl, t_vl, best["batch_size"], shuffle=False)

    model = DeepONetPP(hidden=best["hidden"], p=best["p"], num_heads=best["num_heads"])
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])
    print(f"Final training [{cytokine.upper()}] {grid}x{grid}x{grid}  (max {FULL_EPOCHS} epochs)...")

    t_train_start = time.time()
    train_model(model, opt, ds_tr, ds_vl,
                epochs=FULL_EPOCHS, lam=best["causal_lambda"],
                patience=40, reduce_patience=15, verbose=True)
    train_elapsed = time.time() - t_train_start

    t_pred_start = time.time()
    Yp = predict_full(model, Xbranch, Xtrunk)
    pred_elapsed = time.time() - t_pred_start

    Y_phys  = denormalize(Y,  clip_max)
    Yp_phys = denormalize(Yp, clip_max)

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "model": "DeepONetPP",
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params":          best,
        "optuna_best_val_loss": optuna_val,
        "train_time_seconds":   round(train_elapsed, 2),
        "pred_time_seconds":    round(pred_elapsed,  2),
        "n_parameters":         int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(Y_phys[80:90], Yp_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(Y_phys[90:99], Yp_phys[90:99], M[90:99], clip_max),
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
    ap.add_argument("--out",      type=str, default="./models/deeponet_pp_3d",
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
