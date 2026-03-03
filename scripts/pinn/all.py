"""
pinn.py
=======
Physics-Informed Neural Network surrogate for cytokine field prediction.
Single-cytokine per run — comparable to deeponet_h.py and sta_lstm.py.

Architecture (Fig. 1 — PINN):
  FNN: FC → tanh → FC → tanh → FC → tanh → FC(1)
  Input:  (x, y, t)  — normalised to [-1, 1]
  Output: u — single cytokine concentration (scaled)

Loss = L_data + λ_pde * L_pde + λ_ic * L_ic + λ_bc * L_bc

PDE (reaction-diffusion):
  ∂u/∂t = D*(∂²u/∂x² + ∂²u/∂y²) - k*u + s1(x,y) + s2(x,y) - e(x,y)*u

Usage:
  python scripts/pinn/all.py --grid 50 --cytokine il8 --seed 42
"""

import os, json, argparse, random, warnings, gc
from pathlib import Path

import numpy as np
import tensorflow as tf
import deepxde as dde
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()

# ── Constants ──────────────────────────────────────────────────────────────────
N_TRIALS    = 20
TUNE_ITERS  = 2_000
FULL_ITERS  = 50_000
N_OBS_PER_T = 100
N_DOMAIN    = 500
N_BOUNDARY  = 200
N_INITIAL   = 0

TRUE_SIZE = 5.0
S_MCS     = 60.0
H_MCS     = 1.0 / S_MCS

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
MASK_E, MASK_NDN, MASK_NA, MASK_M1, MASK_M2 = 0, 1, 2, 3, 4


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    dde.config.set_random_seed(seed)


# ── PDE parameters ─────────────────────────────────────────────────────────────

def compute_pde_params(G, cyt_idx):
    """Return scalar D, k, s1_arr, s2_arr, e_arr for single cytokine."""
    areaconv   = TRUE_SIZE**2 / G**2
    volumeconv = TRUE_SIZE**2 / G**2

    D_all = np.array([
        2.09e-6, 3.00e-7, 8.49e-8, 1.45e-8, 4.07e-9, 2.60e-7
    ]) * S_MCS / areaconv

    k_all = np.array([
        0.200, 0.600, 0.500, 0.500, 0.500*0.225, 0.500/25.0
    ]) * H_MCS

    sec = np.array([
        234e-5, 1.46e-5, 3.024e-5,   # il8: keil8, kndnil8, thetanail8
        225e-5,                        # il1: knail1
        250e-5,                        # il6: km1il6
        45e-5,                         # il10: km2il10 (M1)
        250e-5,                        # tnf: knatnf
        70e-5,                         # tnf: km1tnf
        280e-5,                        # tgf: km2tgf
    ]) * volumeconv * H_MCS

    return float(D_all[cyt_idx]), float(k_all[cyt_idx]), sec


def build_source_terms_1cyt(masks_flat, sec, cyt_idx, G):
    """
    Build source term tensors for a single cytokine.
    Returns s1_tf, s2_tf, e_tf each shape (G*G, 1).
    """
    n   = G * G
    me  = masks_flat[:, MASK_E]
    mnn = masks_flat[:, MASK_NDN]
    mna = masks_flat[:, MASK_NA]
    mm1 = masks_flat[:, MASK_M1]
    mm2 = masks_flat[:, MASK_M2]
    z   = np.zeros(n, np.float64)

    # s1, s2, e per cytokine
    s1_map = [sec[0]*me, sec[3]*mna, sec[4]*mm1, sec[5]*mm1, sec[6]*mna, sec[8]*mm2]
    s2_map = [sec[1]*mnn, z, z, z, sec[7]*mm1, z]
    e_map  = [sec[2]*mna, z, z, z, z, z]

    s1 = tf.constant(s1_map[cyt_idx].reshape(-1, 1), dtype=tf.float64)
    s2 = tf.constant(s2_map[cyt_idx].reshape(-1, 1), dtype=tf.float64)
    e  = tf.constant(e_map[cyt_idx].reshape(-1, 1),  dtype=tf.float64)
    return s1, s2, e


# ── PDE residual ───────────────────────────────────────────────────────────────

def make_pde_1cyt(D, k, s1_tf, s2_tf, e_tf, G):
    """PDE for single cytokine. y: (N,1), x: (N,3)."""
    G_tf  = tf.constant(G, dtype=tf.float64)
    D_tf  = tf.constant([[D]], dtype=tf.float64)
    k_tf  = tf.constant([[k]], dtype=tf.float64)

    def pde(x, y):
        ftype  = tf.float64
        G_dyn  = tf.cast(G_tf, ftype)

        d2x = dde.grad.hessian(y, x, i=0, j=0)
        d2y = dde.grad.hessian(y, x, i=1, j=1)
        ut  = dde.grad.jacobian(y, x, i=0, j=2)

        ix = tf.cast(tf.clip_by_value(
            tf.floor((x[:, 0:1] + 1.0) / 2.0 * G_dyn),
            0.0, G_dyn - 1.0), tf.int32)
        iy = tf.cast(tf.clip_by_value(
            tf.floor((x[:, 1:2] + 1.0) / 2.0 * G_dyn),
            0.0, G_dyn - 1.0), tf.int32)
        flat_idx = tf.squeeze(ix * tf.cast(G_tf, tf.int32) + iy, axis=1)

        s1_q = tf.cast(tf.gather(s1_tf, flat_idx), ftype)
        s2_q = tf.cast(tf.gather(s2_tf, flat_idx), ftype)
        e_q  = tf.cast(tf.gather(e_tf,  flat_idx), ftype)

        rhs = D_tf * (d2x + d2y) - k_tf * y + s1_q + s2_q - e_q * y
        return ut - rhs

    return pde


# ── Metrics ────────────────────────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred, masks):
    T  = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
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
        "Global_R2":           r2,
        "Masked_RMSE":         rmse,
        "Avg_Dice":            float(np.mean(dices)),
        "Spatial_Correlation": float(np.mean(corrs))   if corrs  else 0.0,
        "SSIM":                float(np.mean(ssims_v)) if ssims_v else 0.0,
    }

def denormalize(x, clip_max):
    return (np.asarray(x, np.float64) + 1.0) / 2.0 * clip_max


# ── Model builder ──────────────────────────────────────────────────────────────

def build_dde_model(G, hidden, n_layers, lr,
                    pde_fn, geomtime,
                    X_ic, Y_ic_obs,
                    X_obs, Y_obs,
                    bc_fn,
                    lambda_pde=0.01, lambda_ic=1.0, lambda_bc=0.1):
    """Single-cytokine PINN: FNN with 1 output."""

    ic_bc  = dde.PointSetBC(X_ic, Y_ic_obs, component=0)
    neu_bc = dde.NeumannBC(geomtime,
                           lambda x: np.zeros((len(x), 1), np.float64),
                           bc_fn, component=0)
    obs_bc = dde.PointSetBC(X_obs, Y_obs, component=0)

    data = dde.data.TimePDE(
        geomtime, pde_fn,
        ic_bcs=[ic_bc, neu_bc, obs_bc],
        num_domain=N_DOMAIN,
        num_boundary=N_BOUNDARY,
        num_initial=N_INITIAL,
        train_distribution="uniform",
    )

    net = dde.maps.FNN(
        [3] + [hidden] * n_layers + [1],
        "tanh", "Glorot uniform"
    )
    net.apply_output_transform(lambda x, y: y)  # linear — data in [-1,1]

    loss_weights = [lambda_pde, lambda_bc, lambda_ic, 1.0]
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    return model


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_xy_grid(G):
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)

def t_to_norm(t_idx, n_total=101, window=2):
    t_norm_all = np.linspace(-1.0, 1.0, n_total, dtype=np.float64)
    return float(t_norm_all[t_idx + window])


# ── Optuna ─────────────────────────────────────────────────────────────────────

def make_objective(G, cyt_idx, pde_fn, geomtime,
                   X_ic, Y_ic_obs, X_obs_tr, Y_obs_tr,
                   arrays_phys, val_times, bc_fn, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session(); gc.collect()

        hidden   = trial.suggest_categorical("hidden",        [64, 128, 256])
        n_layers = trial.suggest_categorical("n_layers",      [3, 4, 5])
        lr       = trial.suggest_float("learning_rate",       1e-4, 5e-3, log=True)

        model = build_dde_model(
            G, hidden, n_layers, lr,
            pde_fn, geomtime,
            X_ic, Y_ic_obs,
            X_obs_tr, Y_obs_tr,
            bc_fn,
        )
        model.train(iterations=TUNE_ITERS, display_every=TUNE_ITERS)

        xy_grid = make_xy_grid(G)
        mses = []
        for t in val_times:
            tt  = np.full((xy_grid.shape[0], 1), t_to_norm(t), np.float64)
            X_q = np.hstack([xy_grid, tt])
            Yp  = model.predict(X_q)                      # (G*G, 1)
            Yt  = arrays_phys[t, :, :, cyt_idx].reshape(-1, 1).astype(np.float64)
            mses.append(float(np.mean((Yt - Yp)**2)))
        return float(np.mean(mses))
    return objective


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    cyt_idx = CYTOKINE_NAMES.index(cytokine.lower())

    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir   = Path("./models/pinn"); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}×{grid} — loading data...")

    Y_raw  = np.load(data_path/"Y_raw_phys.npy").astype(np.float64)      # (101,G,G,6)
    Y_tgt  = np.load(data_path/"Y_target.npy").astype(np.float64)        # (99,G,G,6)
    M_spat = np.load(data_path/"Y_masks_spatial.npy").astype(np.float64) # (99,G,G,5)
    M_pinn = np.load(data_path/"Y_masks_pinn.npy").astype(np.float64)    # (99,G*G,5)
    Y_ic_raw = np.load(data_path/"Y_ic.npy").astype(np.float64)          # (G,G,6)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(np.array(meta["scaling"]["max"])[cyt_idx])

    G2 = grid * grid
    N  = Y_tgt.shape[0]

    # ── PDE ───────────────────────────────────────────────────────────────────
    D, k, sec = compute_pde_params(grid, cyt_idx)
    masks_mean = M_pinn[:70].mean(axis=0)
    s1_tf, s2_tf, e_tf = build_source_terms_1cyt(masks_mean, sec, cyt_idx, grid)
    pde_fn = make_pde_1cyt(D, k, s1_tf, s2_tf, e_tf, grid)

    t_norm_all = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    t_min = float(t_norm_all[2])
    t_max = float(t_norm_all[-1])

    geom      = dde.geometry.Rectangle([-1.0, -1.0], [1.0, 1.0])
    timedomain= dde.geometry.TimeDomain(t_min, t_max)
    geomtime  = dde.geometry.GeometryXTime(geom, timedomain)

    def bc_fn(x, on_boundary): return on_boundary

    # ── IC ────────────────────────────────────────────────────────────────────
    xy_grid  = make_xy_grid(grid)
    tt0      = np.full((G2, 1), t_min, dtype=np.float64)
    X_ic     = np.hstack([xy_grid, tt0])
    Y_ic_obs = Y_ic_raw[:, :, cyt_idx].reshape(G2, 1).astype(np.float64)

    # ── Observations ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    train_indices = list(range(70))
    val_indices   = list(range(70, 80))

    X_obs_list = []; Y_obs_list = []
    for i in train_indices:
        # prefer points inside cell masks (biologically active regions)
        mask_flat = M_pinn[i]  # (G*G, 5)
        active = np.where(mask_flat.max(axis=-1) > 0)[0]
        if len(active) >= N_OBS_PER_T // 2:
            n_active = N_OBS_PER_T // 2
            n_random = N_OBS_PER_T - n_active
            pts_active = rng.choice(active, size=min(n_active, len(active)), replace=False)
            pts_random = rng.choice(G2,    size=n_random, replace=False)
            pts = np.concatenate([pts_active, pts_random])
        else:
            pts = rng.choice(G2, size=min(N_OBS_PER_T, G2), replace=False)
        t_obs = np.full((len(pts), 1), t_norm_all[i + 2], np.float64)
        X_obs_list.append(np.hstack([xy_grid[pts], t_obs]))
        Y_obs_list.append(Y_tgt[i, :, :, cyt_idx].reshape(G2, 1)[pts])

    X_obs_tr = np.vstack(X_obs_list).astype(np.float64)
    Y_obs_tr = np.vstack(Y_obs_list).astype(np.float64)

    print(f"  Train obs: {X_obs_tr.shape[0]}  |  IC points: {X_ic.shape[0]}  |  Grid: {grid}×{grid}={G2}")

    # ── Optuna ────────────────────────────────────────────────────────────────
    print(f"Optuna: {N_TRIALS} trials × {TUNE_ITERS} iters...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(
        make_objective(grid, cyt_idx, pde_fn, geomtime,
                       X_ic, Y_ic_obs,
                       X_obs_tr, Y_obs_tr,
                       Y_raw,
                       val_times=[i + 2 for i in val_indices],
                       bc_fn=bc_fn, seed=seed),
        n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
    )
    best = study.best_params
    print(f"  Best: {best}  |  val_loss = {study.best_value:.6f}")

    # ── Final training ─────────────────────────────────────────────────────────
    tf.keras.backend.clear_session(); set_seed(seed)
    model = build_dde_model(
        grid, best["hidden"], best["n_layers"], best["learning_rate"],
        pde_fn, geomtime,
        X_ic, Y_ic_obs,
        X_obs_tr, Y_obs_tr,
        bc_fn,
    )
    print(f"Final training [{cytokine.upper()}] {grid}×{grid}  ({FULL_ITERS} iters)...")
    model.train(iterations=FULL_ITERS, display_every=2000)

    # ── Predict ────────────────────────────────────────────────────────────────
    print("Predicting full grid...")
    Yp_all = np.zeros((N, grid, grid, 1), dtype=np.float64)
    for i in range(N):
        tt  = np.full((G2, 1), t_norm_all[i + 2], np.float64)
        X_q = np.hstack([xy_grid, tt])
        Yp_all[i, :, :, 0] = model.predict(X_q).reshape(grid, grid)

    Y_phys  = denormalize(Y_tgt[..., cyt_idx:cyt_idx+1], clip_max)
    Yp_phys = denormalize(Yp_all, clip_max)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params":          best,
        "optuna_best_val_loss": float(study.best_value),
        "results": {
            "Interpolation_72_89":  calculate_metrics(
                Y_phys[70:88], Yp_phys[70:88], M_spat[70:88]),
            "Extrapolation_82_100": calculate_metrics(
                Y_phys[80:99], Yp_phys[80:99], M_spat[80:99]),
        },
    }

    with open(out_dir/f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save(str(out_dir/f"weights_{suffix}"))
    print(f"DONE → models/pinn/res_{suffix}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid",     type=int, required=True)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()
    run_pipeline(args.grid, args.seed, args.cytokine)
