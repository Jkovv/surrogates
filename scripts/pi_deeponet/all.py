import os, json, argparse, random, warnings, gc
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import optuna

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400

TRUE_SIZE   = 5.0
S_MCS       = 60.0
H_MCS       = 1.0 / S_MCS
CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
MASK_E, MASK_NDN, MASK_NA, MASK_M1, MASK_M2 = 0, 1, 2, 3, 4

N_COLLOC    = 2000   # PDE collocation points per training step
N_BC        = 200    # boundary points for BC loss
EVAL_CHUNK  = 4096


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


# PDE 
def compute_pde_params(G, cyt_idx):
    areaconv   = TRUE_SIZE**2 / G**2
    volumeconv = TRUE_SIZE**2 / G**2
    D_all = np.array([
        2.09e-6, 3.00e-7, 8.49e-8, 1.45e-8, 4.07e-9, 2.60e-7
    ]) * S_MCS / areaconv
    k_all = np.array([
        0.200, 0.600, 0.500, 0.500, 0.500 * 0.225, 0.500 / 25.0
    ]) * H_MCS
    sec = np.array([
        234e-5, 1.46e-5, 3.024e-5, 225e-5, 250e-5,
        45e-5, 250e-5, 70e-5, 280e-5,
    ]) * volumeconv * H_MCS
    return float(D_all[cyt_idx]), float(k_all[cyt_idx]), sec


def build_source_arrays(masks_mean, sec, cyt_idx, G, clip_max):
    n = G * G
    me  = masks_mean[:, MASK_E]
    mnn = masks_mean[:, MASK_NDN]
    mna = masks_mean[:, MASK_NA]
    mm1 = masks_mean[:, MASK_M1]
    mm2 = masks_mean[:, MASK_M2]
    z   = np.zeros(n, np.float64)

    s1_map = [sec[0]*me, sec[3]*mna, sec[4]*mm1, sec[5]*mm1, sec[6]*mna, sec[8]*mm2]
    s2_map = [sec[1]*mnn, z, z, z, sec[7]*mm1, z]
    e_map  = [sec[2]*mna, z, z, z, z, z]

    scale = 2.0 / (clip_max + 1e-30)
    s1 = s1_map[cyt_idx] * scale
    s2 = s2_map[cyt_idx] * scale
    e  = e_map[cyt_idx]
    return (tf.constant(s1.reshape(-1, 1), tf.float32),
            tf.constant(s2.reshape(-1, 1), tf.float32),
            tf.constant(e.reshape(-1, 1),  tf.float32))


class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, input_dim=7, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc3 = tf.keras.layers.Dense(p, activation="linear")

    def call(self, x, training=False):
        h = self.fc1(x)
        h = self.fc2(h)
        return self.fc3(h)


class Trunk(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.U   = tf.keras.layers.Dense(hidden, activation="tanh")
        self.V   = tf.keras.layers.Dense(hidden, activation="tanh")
        self.W1a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W1b = tf.keras.layers.Dense(hidden, activation="linear")
        self.W2a = tf.keras.layers.Dense(hidden, activation="relu")
        self.W2b = tf.keras.layers.Dense(hidden, activation="linear")
        self.out = tf.keras.layers.Dense(p, activation="linear")

    def call(self, x):
        u = self.U(x); v = self.V(x)
        h = self.W1b(self.W1a(x)); h = h * u + (1.0 - h) * v
        h = self.W2b(self.W2a(h)); h = h * u + (1.0 - h) * v
        return self.out(h)


class PIDeepONet(tf.keras.Model):
    def __init__(self, hidden, p, branch_dim=7):
        super().__init__()
        self.branch = Branch(hidden, p, input_dim=branch_dim)
        self.trunk  = Trunk(hidden, p)
        self.bias   = self.add_weight(shape=(1,), initializer="zeros",
                                      trainable=True, name="bias")
        self._p = p

    def call_data(self, xb, xt, training=False):
        """
        Data forward pass.
        xb: (batch, D_branch) — branch input
        xt: (batch, n_pts, 3) — trunk input (x, y, t)
        Returns: (batch, n_pts, 1)
        """
        b = self.branch(xb, training=training)         # (batch, p)
        t = self.trunk(xt)                              # (batch, n_pts, p)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias  # (batch, n_pts)
        return tf.expand_dims(r, -1)                    # (batch, n_pts, 1)

    def call_physics_single(self, xb_single, xyt):
        b = self.branch(xb_single, training=True)       # (1, p)
        t = self.trunk(xyt)                              # (N_c, p)
        r = tf.reduce_sum(b[0] * t, axis=-1, keepdims=True) + self.bias
        return r                                         # (N_c, 1)

    def call(self, inputs, training=False):
        return self.call_data(inputs[0], inputs[1], training=training)

def build_branch_inputs(Xb, Xt, cyt_idx):
    N, _, G, _, _ = Xb.shape
    G2 = G * G

    f0 = Xb[:, 0, :, :, cyt_idx]  # (N, G, G) cytokine frame 0
    f1 = Xb[:, 1, :, :, cyt_idx]  # (N, G, G) cytokine frame 1

    masks = Xb[:, 0, :, :, 6:]     # (N, G, G, 5)
    mask_any = (masks.max(axis=-1) > 0.5).astype(np.float32)  # (N, G, G)

    xs = np.linspace(0, 1, G, dtype=np.float32)
    ys = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    out = np.zeros((N, 15), dtype=np.float32)
    for i in range(N):
        ff0 = f0[i]; ff1 = f1[i]; m = mask_any[i]
        na = float(np.sum(m)) + 1e-6

        # Cytokine stats (rescale from [-1,1] to [0,1])
        out[i, 0] = (float(np.max(ff0)) + 1.0) / 2.0
        out[i, 1] = (float(np.mean(ff0)) + 1.0) / 2.0
        out[i, 2] = float(np.std(ff0))
        out[i, 3] = (float(np.max(ff1)) + 1.0) / 2.0
        out[i, 4] = (float(np.mean(ff1)) + 1.0) / 2.0
        out[i, 5] = float(np.std(ff1))

        # Cell geometry
        out[i, 6] = float(np.sum(xx * m) / na)   # centroid_x
        out[i, 7] = float(np.sum(yy * m) / na)   # centroid_y
        out[i, 8] = na / G2                        # extent

        # Per-type cell densities
        for ct in range(5):
            out[i, 9 + ct] = float(np.sum(masks[i, :, :, ct] > 0.5)) / G2

        # Time
        out[i, 14] = float(Xt[i, 0, 2])

    return out

def build_trunk_inputs(Xt):
    return Xt[:, :, :3].astype(np.float32)

def build_dataset(Xbranch, Xtrunk, Yf, batch_size, chunk_size, shuffle=True):
    N, n_pts, _ = Xtrunk.shape
    chunks = list(range(0, n_pts, chunk_size))
    d_branch = Xbranch.shape[1]

    def gen():
        order = np.arange(N)
        if shuffle:
            np.random.shuffle(order)
        for i in order:
            xb = Xbranch[i]
            for s in chunks:
                e = min(s + chunk_size, n_pts); size = e - s
                xt = Xtrunk[i, s:e]; y = Yf[i, s:e]
                if size < chunk_size:
                    pad = chunk_size - size
                    xt = np.concatenate([xt, np.zeros((pad, 3), np.float32)])
                    y  = np.concatenate([y,  np.zeros((pad, 1), np.float32)])
                sz = np.array([size], dtype=np.int32)
                yield (xb, xt, sz), y

    sig = (
        (tf.TensorSpec((d_branch,),       tf.float32),
         tf.TensorSpec((chunk_size, 3),   tf.float32),
         tf.TensorSpec((1,),              tf.int32)),
        tf.TensorSpec((chunk_size, 1),    tf.float32),
    )
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))

def compute_pde_residual(model, xb_single, xyt, D, k, s1_tf, s2_tf, e_tf, G):
    G_f = tf.constant(float(G), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xyt)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xyt)
            u = model.call_physics_single(xb_single, xyt)  # (N_c, 1)

        du = tape1.gradient(u, xyt)       # (N_c, 3)
        u_x = du[:, 0:1]
        u_y = du[:, 1:2]
        u_t = du[:, 2:3]

    du_x_grad = tape2.gradient(u_x, xyt)  # (N_c, 3)
    du_y_grad = tape2.gradient(u_y, xyt)   # (N_c, 3)
    u_xx = du_x_grad[:, 0:1]
    u_yy = du_y_grad[:, 1:2]

    del tape1, tape2

    # Source terms lookup
    ix = tf.cast(tf.clip_by_value(
        tf.floor((xyt[:, 0:1] + 1.0) / 2.0 * G_f), 0.0, G_f - 1.0), tf.int32)
    iy = tf.cast(tf.clip_by_value(
        tf.floor((xyt[:, 1:2] + 1.0) / 2.0 * G_f), 0.0, G_f - 1.0), tf.int32)
    flat_idx = tf.squeeze(ix * tf.cast(G_f, tf.int32) + iy, axis=1)

    s1_q = tf.gather(s1_tf, flat_idx)
    s2_q = tf.gather(s2_tf, flat_idx)
    e_q  = tf.gather(e_tf,  flat_idx)

    rhs = D * (u_xx + u_yy) - k * u + s1_q + s2_q - e_q * u
    residual = u_t - rhs
    return residual

def compute_bc_residual(model, xb_single, xyt_bc):
    xyt_var = tf.Variable(xyt_bc, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xyt_var)
        u = model.call_physics_single(xb_single, xyt_var)
    du = tape.gradient(u, xyt_var)  # (N_bc, 3)
    u_x = du[:, 0:1]
    u_y = du[:, 1:2]

    x_coords = xyt_bc[:, 0]
    y_coords = xyt_bc[:, 1]

    on_x_bnd = tf.cast(tf.abs(tf.abs(x_coords) - 1.0) < 0.01, tf.float32)
    on_y_bnd = tf.cast(tf.abs(tf.abs(y_coords) - 1.0) < 0.01, tf.float32)

    # ∂u/∂n = u_x on x-boundaries, u_y on y-boundaries
    dudn = u_x[:, 0] * on_x_bnd + u_y[:, 0] * on_y_bnd
    return dudn

def sample_collocation(N_c, t_min, t_max):
    xy = np.random.uniform(-1.0, 1.0, (N_c, 2)).astype(np.float32)
    t  = np.random.uniform(t_min, t_max, (N_c, 1)).astype(np.float32)
    return np.hstack([xy, t])

def sample_boundary(N_bc, t_min, t_max):
    pts = []
    n_per_side = N_bc // 4
    for _ in range(n_per_side):
        t = np.random.uniform(t_min, t_max)
        pts.append([-1.0, np.random.uniform(-1, 1), t])  # left
        pts.append([1.0,  np.random.uniform(-1, 1), t])  # right
        pts.append([np.random.uniform(-1, 1), -1.0, t])  # bottom
        pts.append([np.random.uniform(-1, 1),  1.0, t])  # top
    return np.array(pts, dtype=np.float32)[:N_bc]

def masked_mse(pred, y, sz):
    idx  = tf.range(tf.shape(pred)[1])[tf.newaxis, :, tf.newaxis]
    mask = tf.cast(idx < tf.cast(sz[:, tf.newaxis, :], tf.int32), tf.float32)
    return tf.reduce_sum(tf.square(pred - y) * mask) / (tf.reduce_sum(mask) + 1e-8)

def train_step_data(model, opt, xb, xt, sz, y):
    with tf.GradientTape() as tape:
        loss = masked_mse(model.call_data(xb, xt, training=True), y, sz)
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))
    return float(loss)

def train_step_physics(model, opt, xb_single, xyt_colloc, xyt_bc,
                       D, k, s1_tf, s2_tf, e_tf, G,
                       lambda_pde, lambda_bc):

    xyt_var = tf.Variable(xyt_colloc, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # PDE residual
        residual = compute_pde_residual(
            model, xb_single, xyt_var, D, k, s1_tf, s2_tf, e_tf, G)
        loss_pde = lambda_pde * tf.reduce_mean(tf.square(residual))

        # BC residual
        dudn = compute_bc_residual(model, xb_single, xyt_bc)
        loss_bc = lambda_bc * tf.reduce_mean(tf.square(dudn))

        loss_total = loss_pde + loss_bc

    grads = tape.gradient(loss_total, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables)
                      if g is not None]
    if grads_and_vars:
        opt.apply_gradients(grads_and_vars)
    return float(loss_pde), float(loss_bc)


def val_mse(model, ds_vl):
    losses = []
    for batch in ds_vl:
        (xb, xt, sz), y = batch
        pred = model.call_data(xb, xt, training=False)
        losses.append(float(masked_mse(pred, y, sz)))
    return float(np.mean(losses))


def train_model(model, opt, ds_tr, ds_vl,
                Xbranch_tr, t_min, t_max, G,
                D, k, s1_tf, s2_tf, e_tf,
                lambda_pde=0.01, lambda_bc=0.001,
                n_colloc=N_COLLOC, n_bc=N_BC,
                epochs=FULL_EPOCHS, patience=40, reduce_patience=15,
                min_lr=1e-7, verbose=True, physics_every=1):
    best_val = np.inf; best_w = None; wait = rw = 0
    N_tr = Xbranch_tr.shape[0]

    for ep in range(1, epochs + 1):
        # data loss 
        data_losses = []
        for batch in ds_tr:
            (xb, xt, sz), y = batch
            dl = train_step_data(model, opt, xb, xt, sz, y)
            data_losses.append(float(dl))
        tr_data = float(np.mean(data_losses))

        # physics loss
        tr_pde = tr_bc = 0.0
        if lambda_pde > 0 and ep % physics_every == 0:
            idx = np.random.randint(0, N_tr)
            xb_s = tf.constant(Xbranch_tr[idx:idx+1])

            xyt_c  = tf.constant(sample_collocation(n_colloc, t_min, t_max))
            xyt_bc = tf.constant(sample_boundary(n_bc, t_min, t_max))

            tr_pde, tr_bc = train_step_physics(
                model, opt, xb_s, xyt_c, xyt_bc,
                D, k, s1_tf, s2_tf, e_tf, G,
                lambda_pde, lambda_bc)

        # val
        vl_mean = val_mse(model, ds_vl)

        if verbose and ep % 20 == 0:
            print(f"  Ep {ep:4d}  data={tr_data:.5f}  pde={tr_pde:.5f}  "
                  f"bc={tr_bc:.5f}  val={vl_mean:.5f}")

        if vl_mean < best_val:
            best_val = vl_mean; best_w = model.get_weights(); wait = rw = 0
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
        xb = tf.constant(Xbranch[i:i+1])
        for s in range(0, n_pts, chunk):
            e = min(s + chunk, n_pts)
            xt = tf.constant(Xtrunk[i:i+1, s:e])
            out[i, s:e] = model.call_data(xb, xt, training=False).numpy()[0]
    return out


def calculate_metrics(y_true, y_pred, masks):
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]; yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)
    rmse = float(np.sqrt(np.sum(np.square(yt-yp)*ms) / (np.sum(ms)+1e-12)))
    r2   = float(r2_score(yt.flatten(), yp.flatten()))
    dices=[]; corrs=[]; ssims_v=[]
    for t in range(T):
        gt = yt[t,:,:,0]; pr = yp[t,:,:,0]
        thr = 0.05*float(np.max(gt)) if np.max(gt)>0 else 1e-9
        gb=(gt>thr).astype(float); pb=(pr>thr).astype(float)
        dices.append((2*np.sum(gb*pb)+1e-6)/(np.sum(gb)+np.sum(pb)+1e-6))
        if np.std(gt)>1e-12 and np.std(pr)>1e-12:
            corrs.append(float(pearsonr(gt.flatten(),pr.flatten())[0]))
        dr = float(np.max(gt)-np.min(gt))
        if dr>1e-12:
            ssims_v.append(float(ssim(gt,pr,data_range=dr)))
    return {
        "Global_R2": r2, "Masked_RMSE": rmse,
        "Avg_Dice": float(np.mean(dices)),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0,
        "SSIM": float(np.mean(ssims_v)) if ssims_v else 0.0,
    }

def denormalize(x, clip_max):
    return (np.asarray(x, np.float64)+1.0)/2.0*clip_max

def make_objective(Xbr_tr, Xtr_tr, Yf_tr, Xbr_vl, Xtr_vl, Yf_vl,
                   t_min, t_max, G, D, k, s1_tf, s2_tf, e_tf, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        p          = trial.suggest_categorical("p",          [64, 128, 256])
        hidden     = trial.suggest_categorical("hidden",     [128, 256])
        lr         = trial.suggest_float("learning_rate",    1e-5, 1e-3, log=True)
        bs         = trial.suggest_categorical("batch_size", [4, 8])
        chunk_size = trial.suggest_categorical("chunk_size", [2048, 4096])
        lambda_pde = trial.suggest_float("lambda_pde",      1e-4, 1.0, log=True)
        lambda_bc  = trial.suggest_float("lambda_bc",       1e-4, 0.1, log=True)

        ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr, bs, chunk_size, shuffle=True)
        ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl, bs, chunk_size, shuffle=False)

        d_branch = Xbr_tr.shape[1]
        model = PIDeepONet(hidden=hidden, p=p, branch_dim=d_branch)
        opt   = tf.keras.optimizers.Adam(lr)
        best  = train_model(
            model, opt, ds_tr, ds_vl,
            Xbr_tr, t_min, t_max, G,
            D, k, s1_tf, s2_tf, e_tf,
            lambda_pde=lambda_pde, lambda_bc=lambda_bc,
            n_colloc=500, n_bc=50,   
            epochs=TUNE_EPOCHS, patience=8,
            reduce_patience=5, verbose=False,
            physics_every=2,
        )
        return float(best)
    return objective


def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    idx = CYTOKINE_NAMES.index(cytokine.lower())

    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir   = Path("./models/pi_deeponet"); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid} — PI-DeepONet — loading data...")
    Xb = np.load(data_path/"X_branch.npy").astype(np.float32)
    Xt = np.load(data_path/"X_trunk.npy").astype(np.float32)
    Y  = np.load(data_path/"Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float32)
    M_pinn = np.load(data_path/"Y_masks_pinn.npy").astype(np.float64)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    N = Xb.shape[0]; G2 = Xt.shape[1]; G = int(round(G2**0.5))
    Yf = Y.reshape(N, G2, 1)

    Xbranch = build_branch_inputs(Xb, Xt, idx)   # (N, 15)
    Xtrunk  = build_trunk_inputs(Xt)  # (N, G*G, 3)

    d_branch = Xbranch.shape[1]
    print(f"  Branch: (N, {d_branch})  |  Trunk: (N, {G2}, 3)  |  clip_max: {clip_max:.6f}")

    # PDE 
    D, k, sec = compute_pde_params(grid, idx)
    masks_mean = M_pinn[:70].mean(axis=0)
    s1_tf, s2_tf, e_tf = build_source_arrays(masks_mean, sec, idx, grid, clip_max)

    # time domain
    t_norm_all = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    t_min = float(t_norm_all[2])
    t_max = float(t_norm_all[-1])

    # 70/10/20
    Xbr_tr, Xtr_tr, Yf_tr = Xbranch[:70],   Xtrunk[:70],   Yf[:70]
    Xbr_vl, Xtr_vl, Yf_vl = Xbranch[70:80], Xtrunk[70:80], Yf[70:80]

    # optuna 
    print(f"Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        make_objective(Xbr_tr, Xtr_tr, Yf_tr, Xbr_vl, Xtr_vl, Yf_vl,
                       t_min, t_max, G, D, k, s1_tf, s2_tf, e_tf, seed),
        n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
    )
    best = study.best_params
    print(f"  Best: {best}  |  val_loss = {study.best_value:.6f}")

    # final training 
    tf.keras.backend.clear_session(); set_seed(seed)
    ds_tr = build_dataset(Xbr_tr, Xtr_tr, Yf_tr,
                          best["batch_size"], best["chunk_size"], shuffle=True)
    ds_vl = build_dataset(Xbr_vl, Xtr_vl, Yf_vl,
                          best["batch_size"], best["chunk_size"], shuffle=False)

    model = PIDeepONet(hidden=best["hidden"], p=best["p"], branch_dim=d_branch)
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])

    print(f"Final training (max {FULL_EPOCHS} epochs, "
          f"λ_pde={best['lambda_pde']:.4f}, λ_bc={best['lambda_bc']:.4f})...")
    train_model(
        model, opt, ds_tr, ds_vl,
        Xbr_tr, t_min, t_max, G,
        D, k, s1_tf, s2_tf, e_tf,
        lambda_pde=best["lambda_pde"], lambda_bc=best["lambda_bc"],
        n_colloc=N_COLLOC, n_bc=N_BC,
        epochs=FULL_EPOCHS, patience=40, reduce_patience=15,
        verbose=True, physics_every=1,
    )

    # evaluate 
    Yp_flat = predict_full(model, Xbranch, Xtrunk)
    Yp      = Yp_flat.reshape(N, G, G, 1)
    Y_phys  = denormalize(Y.reshape(N, G, G, 1), clip_max)
    Yp_phys = denormalize(Yp, clip_max)

    suffix  = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params": best,
        "optuna_best_val_loss": float(study.best_value),
        "results": {
            "Interpolation_72_89": calculate_metrics(
                Y_phys[70:88], Yp_phys[70:88], M[70:88]),
            "Extrapolation_82_100": calculate_metrics(
                Y_phys[80:99], Yp_phys[80:99], M[80:99]),
        },
    }
    with open(out_dir/f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir/f"weights_{suffix}.weights.h5")
    print(f"DONE → models/pi_deeponet/res_{suffix}.json")

    for split, m in results["results"].items():
        print(f"  {split}: R²={m['Global_R2']:.4f}  RMSE={m['Masked_RMSE']:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid",     type=int, default=None)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    if args.grid:
        run_pipeline(args.grid, args.seed, args.cytokine)
    else:
        for d in sorted(Path("./preprocessed").iterdir()):
            if d.is_dir():
                run_pipeline(int(d.name.split("x")[0]), args.seed, args.cytokine)
