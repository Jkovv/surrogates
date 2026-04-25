"""
DeepONet++ (3D) — architectural + optimisation upgrades over baseline DeepONet
for 3D cytokine / reaction-diffusion surrogate modelling.

SCIENTIFIC FAIRNESS CONSTRAINT
------------------------------
The branch input is kept IDENTICAL to the vanilla baseline DeepONet:
8 hand-crafted scalars summarising the t=0 frame (max, mean, std,
centroid_xyz, extent, t_norm). This ensures DeepONet++ and baseline
condition on the same information — only the processing differs.

What DeepONet++ changes RELATIVE TO BASELINE, keeping identical inputs:

  1. Factorized Axial Attention (FAA) in the trunk
     The baseline trunk was a pointwise MLP — every voxel processed in
     isolation. FAA applies attention along x, y, z separately, exposing
     inter-slice coupling explicitly. Cost: O(G²) per axis instead of
     O(G⁶) for full 3D attention. Trunk inputs are unchanged.

  2. Linear–Nonlinear Fusion (LNF) output head
     Reaction-diffusion PDEs have two components: a linear diffusion
     operator (smooth) and a nonlinear reaction term (sharp). LNF uses
     two trunk branches (linear + ReLU-nonlinear), fused by elementwise
     product. Matches the PDE structure.

  3. Causal time-weighted loss
     Upweights early timesteps via exp(-λ·(t_norm+1)). Reduces rollout-
     error accumulation across the 99-frame horizon (PDE-Refiner idea).

  4. Optimisation-only changes
     - bfloat16 mixed precision on A100 tensor cores (~2× speedup)
     - @tf.function JIT compilation bound per-trial
     - GPU-preloaded dataset via from_tensor_slices (no per-batch H2D)
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
# GPU / mixed precision setup (A100)
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
                print(f"  [gpu] mixed precision: mixed_bfloat16 (A100)")
            except Exception:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        print(f"  [gpu] NO GPU DETECTED — running on CPU (slow).")


configure_gpu()

N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


# ----------------------------------------------------------------------
# VANILLA branch — 8 scalars, IDENTICAL to deeponet_3d.py baseline.
# Same architecture (2-layer MLP) and same input features. This is the
# key constraint: DeepONet++ must not peek at extra information.
# ----------------------------------------------------------------------
class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p,      activation="linear")

    def call(self, x, training=False):
        return self.fc2(self.fc1(x))


# ----------------------------------------------------------------------
# Factorized Axial Attention
# ----------------------------------------------------------------------
class AxialAttention(tf.keras.layers.Layer):
    """
    Attention along a single spatial axis of a 3D volume.
    Input: (B, G, G, G, C). Output: same shape.
    """
    def __init__(self, dim, num_heads=4, axis=1, **kw):
        super().__init__(**kw)
        self.axis = axis
        self.mha  = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        shape = tf.shape(x)
        B = shape[0]; Gx = shape[1]; Gy = shape[2]; Gz = shape[3]
        C = x.shape[-1]
        if self.axis == 1:
            perm = [0, 2, 3, 1, 4]; inv = [0, 3, 1, 2, 4]; L = Gx; other = (Gy, Gz)
        elif self.axis == 2:
            perm = [0, 1, 3, 2, 4]; inv = [0, 1, 3, 2, 4]; L = Gy; other = (Gx, Gz)
        else:
            perm = [0, 1, 2, 3, 4]; inv = [0, 1, 2, 3, 4]; L = Gz; other = (Gx, Gy)

        x_p  = tf.transpose(x, perm)
        x_r  = tf.reshape(x_p, (B * other[0] * other[1], L, C))
        attn = self.mha(x_r, x_r)
        x_r  = self.norm(x_r + attn)
        x_p2 = tf.reshape(x_r, (B, other[0], other[1], L, C))
        return tf.transpose(x_p2, inv)


class FAATrunk(tf.keras.layers.Layer):
    """
    Trunk with factorized axial attention. Input features are the SAME as
    the baseline's trunk input (x, y, z + 22 channel values per voxel), just
    laid out as a volume (B, G, G, G, 25) so attention ops can be applied.

    If ablate_z=True, the z-axis attention is skipped (replaced by identity).
    Lets us measure the empirical contribution of inter-slice (z-direction)
    coupling — the direct answer to the supervisor's question
    "how are these slices connected?".
    """
    def __init__(self, dim, p, num_heads=4, ablate_z=False, **kw):
        super().__init__(**kw)
        self.ablate_z = ablate_z
        self.project_in = tf.keras.layers.Dense(dim, activation="gelu")
        self.attn_x = AxialAttention(dim, num_heads=num_heads, axis=1)
        self.attn_y = AxialAttention(dim, num_heads=num_heads, axis=2)
        self.attn_z = AxialAttention(dim, num_heads=num_heads, axis=3)
        self.mlp    = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation="gelu"),
            tf.keras.layers.Dense(p),
        ])

    def call(self, x):
        h = self.project_in(x)
        h = self.attn_x(h); h = self.attn_y(h)
        if not self.ablate_z:
            h = self.attn_z(h)
        return self.mlp(h)


# ----------------------------------------------------------------------
# DeepONet++ = vanilla branch + FAA trunk + LNF output head
# ----------------------------------------------------------------------
class DeepONetPP(tf.keras.Model):
    """
    u(x,y,z) = bias + ⟨branch, trunk_L⟩ · ReLU(⟨branch, trunk_N⟩)

    Branch  : 8 scalars → (B, p)                        — SAME AS BASELINE
    Trunk_L : (B, G, G, G, 25) → (B, G, G, G, p)         — FAA, linear
    Trunk_N : (B, G, G, G, 25) → (B, G, G, G, p)         — FAA, nonlinear

    ablate_z : if True, both trunk's z-axis attention is replaced by identity.
               Used for measuring the slice-coupling contribution (ablation
               study for supervisor's question "how are slices connected?").
    """
    def __init__(self, hidden, p, num_heads=4, ablate_z=False):
        super().__init__()
        self.branch  = Branch(hidden, p)
        self.trunk_L = FAATrunk(dim=hidden, p=p, num_heads=num_heads, ablate_z=ablate_z)
        self.trunk_N = FAATrunk(dim=hidden, p=p, num_heads=num_heads, ablate_z=ablate_z)
        self.bias    = self.add_weight(
            shape=(1,), initializer="zeros", trainable=True, name="bias")
        self.p = p
        self.ablate_z = ablate_z

    def call(self, inputs, training=False):
        xb, xt = inputs                                    # xb: (B, 8), xt: (B,G,G,G,25)
        b  = self.branch(xb, training=training)            # (B, p)
        tL = self.trunk_L(xt)                              # (B, G, G, G, p)
        tN = tf.nn.relu(self.trunk_N(xt))                  # (B, G, G, G, p)
        b_exp = b[:, None, None, None, :]                  # (B, 1, 1, 1, p)
        linear_part    = tf.reduce_sum(b_exp * tL, axis=-1, keepdims=True)
        nonlinear_part = tf.reduce_sum(b_exp * tN, axis=-1, keepdims=True)
        out = linear_part * nonlinear_part + self.bias     # (B, G, G, G, 1)
        return tf.cast(out, tf.float32)                    # cast-back under bf16


# ----------------------------------------------------------------------
# Input builders — IDENTICAL to baseline
# ----------------------------------------------------------------------
def build_branch_inputs(Xb, Xt, cyt_idx):
    """
    8 scalar features per sample, IDENTICAL to baseline deeponet_3d.py:
      0: max_f0     ∈ [0,1]   (rescaled from [-1,1])
      1: mean_f0    ∈ [0,1]
      2: std_f0     ∈ [0,1]
      3: centroid_x ∈ [0,1]
      4: centroid_y ∈ [0,1]
      5: centroid_z ∈ [0,1]
      6: extent     ∈ [0,1]
      7: t_norm     ∈ [-1,1]
    """
    N, _, G, _, _, _ = Xb.shape
    f0   = Xb[:, 0, :, :, :, cyt_idx]
    mask = (Xb[:, 0, :, :, :, 6:].max(axis=-1) > 0.5).astype(np.float32)
    xs = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, xs, xs, indexing='ij')
    out = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        f = f0[i]; m = mask[i]; na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(f))  + 1.0) / 2.0
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0
        out[i, 2] = float(np.std(f))
        out[i, 3] = float(np.sum(xx * m) / na)
        out[i, 4] = float(np.sum(yy * m) / na)
        out[i, 5] = float(np.sum(zz * m) / na)
        out[i, 6] = na / (G * G * G)
        out[i, 7] = float(Xt[i, 0, 3])
    return out


def build_trunk_inputs_volumetric(Xb, Xt):
    """
    Per-voxel trunk features: (x, y, z) + 22 stacked channel values from the
    2-frame window. IDENTICAL content to the baseline's trunk input — only
    the layout differs: baseline uses (N, G³, 25) flat; FAA needs the volume
    shape (N, G, G, G, 25) so attention can factor along axes.
    """
    N, _, G, _, _, _ = Xb.shape
    xyz_flat = Xt[:, :, :3].astype(np.float32)
    xyz_vol  = xyz_flat.reshape(N, G, G, G, 3)
    vals     = Xb.transpose(0, 2, 3, 4, 1, 5).reshape(N, G, G, G, 22).astype(np.float32)
    return np.concatenate([xyz_vol, vals], axis=-1)        # (N, G, G, G, 25)


# ----------------------------------------------------------------------
# Dataset — GPU-preloaded tensors
# ----------------------------------------------------------------------
def build_dataset(Xbranch, Xtrunk, Y_vol, t_norm, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        ((Xbranch.astype(np.float32),
          Xtrunk.astype(np.float32),
          t_norm.astype(np.float32)),
         Y_vol.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(len(Xbranch), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------
# Causal time-weighted loss + JIT-compiled train/val steps
# ----------------------------------------------------------------------
def _causal_se(pred, y, t_norm, lam):
    w  = tf.exp(-lam * (t_norm + 1.0))
    se = tf.square(pred - y)
    return tf.reduce_mean(w[:, None, None, None, None] * se)


def _make_steps(model, opt):
    """
    tf.function pair bound to a SPECIFIC model+optimizer. Each Optuna trial
    makes its own pair so we never re-trace across model instances.
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
        return _causal_se(model([xb, xt], training=False), y, t_norm, lam)

    return train_step, val_step


def train_model(model, opt, ds_tr, ds_vl,
                epochs, lam=0.5, patience=40, reduce_patience=15,
                min_lr=1e-7, verbose=True):
    # Eager warmup so all tf.Variables exist BEFORE any @tf.function call.
    for (xb, xt, _t), _y in ds_tr.take(1):
        _ = model([xb, xt], training=False)
        break

    train_step_fn, val_step_fn = _make_steps(model, opt)

    best_val = np.inf; best_w = None; wait = rw = 0
    for ep in range(1, epochs + 1):
        tr_losses = [float(train_step_fn(xb, xt, y, t_b, lam))
                     for (xb, xt, t_b), y in ds_tr]
        vl_losses = [float(val_step_fn(xb, xt, y, t_b, lam))
                     for (xb, xt, t_b), y in ds_vl]
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
    N = Xtrunk.shape[0]; G = Xtrunk.shape[1]
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


def compute_2d_slice_metrics(yt, yp, clip_max):
    """
    Per-axis 2D mid-slice metrics — addresses the supervisor's question
    "in 3D, only 2D slices matter, biologists only see 2D slices".
    For each of the three orthogonal mid-planes (xy at z=G/2, xz at y=G/2,
    yz at x=G/2) we compute R² and SSIM averaged across T time steps.

    yt, yp : (T, G, G, G, 1)
    """
    T = yt.shape[0]; G = yt.shape[1]
    fixed_dr = float(clip_max) if clip_max > 0 else 1.0
    mid = G // 2

    out = {}
    for axis_name, sl in (("xy_midplane_z",  np.s_[:, :, :, mid, 0]),
                          ("xz_midplane_y",  np.s_[:, :, mid, :, 0]),
                          ("yz_midplane_x",  np.s_[:, mid, :, :, 0])):
        gts = yt[sl]; prs = yp[sl]
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
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]; yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)

    sq_diff = np.square(yt - yp)
    rmse = float(np.sqrt(np.sum(sq_diff * ms) / (np.sum(ms) + 1e-12)))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))
    r2 = float(r2_score(yt.flatten(), yp.flatten()))

    per_t_r2 = []
    for t in range(T):
        gt_f = yt[t].flatten(); pr_f = yp[t].flatten()
        per_t_r2.append(float(r2_score(gt_f, pr_f))
                        if np.std(gt_f) > 1e-12 else np.nan)

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
            dices.append((2.0 * np.sum(gb * pb)) /
                         (np.sum(gb) + np.sum(pb) + 1e-12))
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
        "Slice_2D":            compute_2d_slice_metrics(yt, yp, clip_max),
    }


def denormalize(x, clip_max):
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * clip_max


# ----------------------------------------------------------------------
# Optuna
# ----------------------------------------------------------------------
def make_objective(Xbr_tr, Xtr_tr, Y_tr, t_tr,
                   Xbr_vl, Xtr_vl, Y_vl, t_vl, seed, ablate_z=False):
    def objective(trial):
        set_seed(seed); tf.keras.backend.clear_session()
        p         = trial.suggest_categorical("p",          [32, 64, 128])
        hidden    = trial.suggest_categorical("hidden",     [64, 128])
        num_heads = trial.suggest_categorical("num_heads",  [2, 4])
        lr        = trial.suggest_float      ("learning_rate", 1e-5, 1e-3, log=True)
        bs        = trial.suggest_categorical("batch_size", [1, 2])
        lam       = trial.suggest_float      ("causal_lambda", 0.0, 1.0)

        ds_tr = build_dataset(Xbr_tr, Xtr_tr, Y_tr, t_tr, bs, shuffle=True)
        ds_vl = build_dataset(Xbr_vl, Xtr_vl, Y_vl, t_vl, bs, shuffle=False)

        model = DeepONetPP(hidden=hidden, p=p, num_heads=num_heads, ablate_z=ablate_z)
        opt   = tf.keras.optimizers.Adam(lr)
        best  = train_model(model, opt, ds_tr, ds_vl,
                            epochs=TUNE_EPOCHS, lam=lam,
                            patience=8, reduce_patience=5, verbose=False)
        return float(best)
    return objective


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
# Fixed time-series split over the 99 prediction samples:
#   train : indices  0..69    (abs t=2..71)
#   val   : indices 70..79    (abs t=72..81)
#   test  : indices 80..98    (abs t=82..100)
#     near-horizon window : indices 80..89  (abs t=82..91)   — 10 frames
#     far-horizon  window : indices 90..98  (abs t=92..100)  —  9 frames
def run_pipeline(grid, seed, cytokine,
                 data_root="./preprocessed_3d",
                 out_root="./models/deeponet_pp_3d",
                 ablate_z=False):
    set_seed(seed)
    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx = cyt_names.index(cytokine.lower())

    data_path = Path(f"{data_root}/{grid}x{grid}x{grid}")
    out_dir   = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid}x{grid} — loading data...")
    if ablate_z:
        print(f"  *** ABLATION MODE: z-axis attention DISABLED ***")
    Xb = np.load(data_path/"X_branch.npy").astype(np.float32)
    Xt = np.load(data_path/"X_trunk.npy").astype(np.float32)
    Y  = np.load(data_path/"Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float32)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    N = Xb.shape[0]; G = Xb.shape[2]
    t_all = Xt[:, 0, 3].astype(np.float32)

    # VANILLA branch (8 scalars) — identical to baseline
    Xbranch = build_branch_inputs(Xb, Xt, idx)             # (N, 8)
    Xtrunk  = build_trunk_inputs_volumetric(Xb, Xt)        # (N, G, G, G, 25)

    print(f"  Branch  : (N, 8)                 — VANILLA, same as baseline")
    print(f"  Trunk   : (N, {G}, {G}, {G}, 25)   — FAA, same features as baseline")

    # Fixed 70/10/20 split
    Xbr_tr, Xtr_tr, Y_tr, t_tr = Xbranch[:70],   Xtrunk[:70],   Y[:70],   t_all[:70]
    Xbr_vl, Xtr_vl, Y_vl, t_vl = Xbranch[70:80], Xtrunk[70:80], Y[70:80], t_all[70:80]

    # When ablating, ALWAYS reuse the HP from the corresponding non-ablated
    # seed-42 run. This is what makes the ablation a fair comparison: same
    # architecture HP, only the z-attention path is changed.
    if ablate_z:
        ref_path = out_dir / f"res_{cytokine}_{grid}_42.json"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Ablation needs the non-ablated reference run first: {ref_path}")
        print(f"  [ablate] Reusing HP from {ref_path.name}")
        with open(ref_path) as f:
            ref = json.load(f)
        best = ref["best_params"]
        optuna_val = ref["optuna_best_val_loss"]
    elif seed == 42:
        print(f"Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(
            make_objective(Xbr_tr, Xtr_tr, Y_tr, t_tr,
                           Xbr_vl, Xtr_vl, Y_vl, t_vl, 42, ablate_z=False),
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

    ds_tr = build_dataset(Xbr_tr, Xtr_tr, Y_tr, t_tr, best["batch_size"], shuffle=True)
    ds_vl = build_dataset(Xbr_vl, Xtr_vl, Y_vl, t_vl, best["batch_size"], shuffle=False)

    model = DeepONetPP(hidden=best["hidden"], p=best["p"],
                       num_heads=best["num_heads"], ablate_z=ablate_z)
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])
    print(f"Final training [{cytokine.upper()}] {grid}x{grid}x{grid} "
          f"(max {FULL_EPOCHS} epochs)...")

    t_train_start = time.time()
    train_model(model, opt, ds_tr, ds_vl,
                epochs=FULL_EPOCHS, lam=best["causal_lambda"],
                patience=40, reduce_patience=15, verbose=True)
    train_elapsed = time.time() - t_train_start
    print(f"  Training time: {train_elapsed:.1f}s")

    t_pred_start = time.time()
    Yp = predict_full(model, Xbranch, Xtrunk)
    pred_elapsed = time.time() - t_pred_start
    print(f"  Prediction time (all {N} samples): {pred_elapsed:.1f}s")

    Y_phys  = denormalize(Y,  clip_max)
    Yp_phys = denormalize(Yp, clip_max)

    # Naming: res_<cytokine>_<grid>_<seed>.json + matching .weights.h5
    # Ablation runs get an extra suffix so they don't overwrite the main run.
    suffix = f"{cytokine}_{grid}_{seed}" + ("_ablate_z" if ablate_z else "")

    results = {
        "model":      "DeepONetPP_vanilla_branch" + ("_ablate_z" if ablate_z else ""),
        "grid":       grid,
        "seed":       seed,
        "cytokine":   cytokine,
        "ablate_z":   bool(ablate_z),
        "best_params":          best,
        "optuna_best_val_loss": optuna_val,
        "train_time_seconds":   round(train_elapsed, 2),
        "pred_time_seconds":    round(pred_elapsed,  2),
        "n_parameters": int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Y_phys[80:90], Yp_phys[80:90], M[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Y_phys[90:99], Yp_phys[90:99], M[90:99], clip_max),
        },
    }
    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    print(f"DONE → {out_dir}/res_{suffix}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid",     type=int, default=None)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--data",     type=str, default="./preprocessed_3d")
    ap.add_argument("--out",      type=str, default="./models/deeponet_pp_3d")
    ap.add_argument("--ablate-z-attention", action="store_true",
                    help="Disable z-axis attention (replaces it with identity). "
                         "Reuses HP from the corresponding non-ablated seed-42 "
                         "run, so requires that run to exist first. Output "
                         "files get an '_ablate_z' suffix.")
    args = ap.parse_args()

    if args.grid:
        run_pipeline(args.grid, args.seed, args.cytokine,
                     data_root=args.data, out_root=args.out,
                     ablate_z=args.ablate_z_attention)
    else:
        for d in sorted(Path(args.data).iterdir()):
            if d.is_dir():
                run_pipeline(int(d.name.split("x")[0]), args.seed, args.cytokine,
                             data_root=args.data, out_root=args.out,
                             ablate_z=args.ablate_z_attention)
