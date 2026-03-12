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

N_TRIALS    = 20
TUNE_ITERS  = 2_000
FULL_ITERS  = 15_000
LBFGS_ITERS = 5_000     
N_OBS_PER_T = 100
N_DOMAIN    = 5_000
N_BOUNDARY  = 1_000
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
        234e-5,    # [0] keil8
        1.46e-5,   # [1] kndnil8
        3.024e-5,  # [2] thetanail8
        225e-5,    # [3] knail1
        250e-5,    # [4] km1il6
        45e-5,     # [5] km2il10
        250e-5,    # [6] knatnf
        70e-5,     # [7] km1tnf
        280e-5,    # [8] km2tgf
    ]) * volumeconv * H_MCS

    return float(D_all[cyt_idx]), float(k_all[cyt_idx]), sec


def build_source_terms_1cyt(masks_flat, sec, cyt_idx, G, clip_max):
    n   = G * G
    me  = masks_flat[:, MASK_E]
    mnn = masks_flat[:, MASK_NDN]
    mna = masks_flat[:, MASK_NA]
    mm1 = masks_flat[:, MASK_M1]
    mm2 = masks_flat[:, MASK_M2]
    z   = np.zeros(n, np.float64)

    # sec[5] = km2il10 → M2 macrophage secretion for IL-10
    s1_map = [sec[0]*me, sec[3]*mna, sec[4]*mm1, sec[5]*mm2, sec[6]*mna, sec[8]*mm2]
    s2_map = [sec[1]*mnn, z, z, z, sec[7]*mm1, z]
    e_map  = [sec[2]*mna, z, z, z, z, z]

    scale_factor = 2.0 / (clip_max + 1e-30)

    s1_arr = s1_map[cyt_idx].reshape(-1, 1) * scale_factor
    s2_arr = s2_map[cyt_idx].reshape(-1, 1) * scale_factor
    e_arr  = e_map[cyt_idx].reshape(-1, 1) 

    s1_tf = tf.constant(s1_arr, dtype=tf.float64)
    s2_tf = tf.constant(s2_arr, dtype=tf.float64)
    e_tf  = tf.constant(e_arr,  dtype=tf.float64)
    return s1_tf, s2_tf, e_tf


# PDE residual 
def make_pde_1cyt(D, k, s1_tf, s2_tf, e_tf, G):
    G_tf = tf.constant(G, dtype=tf.float64)
    D_tf = tf.constant([[D]], dtype=tf.float64)
    k_tf = tf.constant([[k]], dtype=tf.float64)

    def pde(x, y):
        d2x = dde.grad.hessian(y, x, i=0, j=0)
        d2y = dde.grad.hessian(y, x, i=1, j=1)
        ut  = dde.grad.jacobian(y, x, i=0, j=2)

        G_dyn = tf.cast(G_tf, tf.float64)
        ix = tf.cast(tf.clip_by_value(
            tf.floor((x[:, 0:1] + 1.0) / 2.0 * G_dyn),
            0.0, G_dyn - 1.0), tf.int32)
        iy = tf.cast(tf.clip_by_value(
            tf.floor((x[:, 1:2] + 1.0) / 2.0 * G_dyn),
            0.0, G_dyn - 1.0), tf.int32)
        flat_idx = tf.squeeze(ix * tf.cast(G_tf, tf.int32) + iy, axis=1)

        s1_q = tf.gather(s1_tf, flat_idx)
        s2_q = tf.gather(s2_tf, flat_idx)
        e_q  = tf.gather(e_tf,  flat_idx)

        # ∂u/∂t = D·Δu − k·u + s1_scaled + s2_scaled − e·u
        rhs = D_tf * (d2x + d2y) - k_tf * y + s1_q + s2_q - e_q * y
        return ut - rhs

    return pde


# metrics 
def _fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def _inv_fisher_z(z):
    return float(np.tanh(z))

def calculate_metrics(y_true, y_pred, masks, clip_max):
    T  = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
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

def build_dde_model(G, hidden, n_layers, lr,
                    pde_fn, geomtime,
                    X_ic, Y_ic_obs,
                    X_obs, Y_obs,
                    bc_fn,
                    lambda_pde=0.01, lambda_ic=1.0, lambda_bc=0.1):

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
    net.apply_output_transform(lambda x, y: y)

    # loss order: [PDE, ic_bc, neu_bc, obs_bc]
    loss_weights = [lambda_pde, lambda_ic, lambda_bc, 1.0]
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    return model

def make_xy_grid(G):
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def t_to_norm(t_idx, n_total=101):
    t_norm_all = np.linspace(-1.0, 1.0, n_total, dtype=np.float64)
    return float(t_norm_all[t_idx])

def make_objective(G, cyt_idx, pde_fn, geomtime,
                   X_ic, Y_ic_obs, X_obs_tr, Y_obs_tr,
                   Y_tgt_scaled, val_indices, bc_fn, seed):

    xy_grid = make_xy_grid(G)
    G2 = G * G

    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session(); gc.collect()

        hidden     = trial.suggest_categorical("hidden",        [50, 64, 128])
        n_layers   = trial.suggest_categorical("n_layers",      [3, 4])
        lr         = trial.suggest_float("learning_rate",       1e-4, 5e-3, log=True)
        lambda_pde = trial.suggest_float("lambda_pde",         1e-3, 1.0, log=True)
        lambda_ic  = trial.suggest_float("lambda_ic",          0.1,  10.0, log=True)
        lambda_bc  = trial.suggest_float("lambda_bc",          1e-3, 1.0, log=True)

        model = build_dde_model(
            G, hidden, n_layers, lr,
            pde_fn, geomtime,
            X_ic, Y_ic_obs,
            X_obs_tr, Y_obs_tr,
            bc_fn,
            lambda_pde=lambda_pde,
            lambda_ic=lambda_ic,
            lambda_bc=lambda_bc,
        )
        model.train(iterations=TUNE_ITERS, display_every=TUNE_ITERS + 1)

        mses = []
        for vi in val_indices:
            abs_t = vi + 2  
            tt  = np.full((G2, 1), t_to_norm(abs_t), np.float64)
            X_q = np.hstack([xy_grid, tt])
            Yp  = model.predict(X_q) 
            Yt  = Y_tgt_scaled[vi, :, :, cyt_idx].reshape(-1, 1).astype(np.float64)
            mses.append(float(np.mean((Yt - Yp) ** 2)))
        return float(np.mean(mses))

    return objective

def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    cyt_idx = CYTOKINE_NAMES.index(cytokine.lower())

    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir   = Path("./models/pinn"); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid} — loading data...")

    Y_tgt  = np.load(data_path / "Y_target.npy").astype(np.float64)         # (99,G,G,6) scaled
    M_spat = np.load(data_path / "Y_masks_spatial.npy").astype(np.float64)   # (99,G,G,5)
    M_pinn = np.load(data_path / "Y_masks_pinn.npy").astype(np.float64)     # (99,G*G,5)
    Y_ic_full = np.load(data_path / "Y_ic.npy").astype(np.float64)          # (G,G,6) scaled

    with open(data_path / "metadata.json") as f:
        meta = json.load(f)
    clip_max = float(np.array(meta["scaling"]["max"])[cyt_idx])

    G2 = grid * grid
    N  = Y_tgt.shape[0]  # 99

    D, k, sec = compute_pde_params(grid, cyt_idx)

    masks_mean = M_pinn[:70].mean(axis=0)  # (G*G, 5)
    s1_tf, s2_tf, e_tf = build_source_terms_1cyt(masks_mean, sec, cyt_idx, grid, clip_max)
    pde_fn = make_pde_1cyt(D, k, s1_tf, s2_tf, e_tf, grid)

    t_norm_all = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    t_min = float(t_norm_all[2])   # first prediction timestep
    t_max = float(t_norm_all[-1])  # last timestep

    geom      = dde.geometry.Rectangle([-1.0, -1.0], [1.0, 1.0])
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    geomtime  = dde.geometry.GeometryXTime(geom, timedomain)

    def bc_fn(x, on_boundary):
        return on_boundary

    xy_grid  = make_xy_grid(grid)
    tt0      = np.full((G2, 1), t_min, dtype=np.float64)
    X_ic     = np.hstack([xy_grid, tt0])
    Y_ic_obs = Y_ic_full[:, :, cyt_idx].reshape(G2, 1).astype(np.float64)  # scaled

    rng = np.random.default_rng(seed)
    train_indices = list(range(70))
    val_indices   = list(range(70, 80))

    X_obs_list = []; Y_obs_list = []
    for i in train_indices:
        abs_t = i + 2  # absolute timestep
        t_obs = np.full((G2, 1), t_norm_all[abs_t], np.float64)
        X_obs_list.append(np.hstack([xy_grid, t_obs]))
        Y_obs_list.append(
            Y_tgt[i, :, :, cyt_idx].reshape(G2, 1).astype(np.float64)
        )

    X_obs_tr = np.vstack(X_obs_list)
    Y_obs_tr = np.vstack(Y_obs_list)

    print(f"  Train obs: {X_obs_tr.shape[0]}  |  IC points: {X_ic.shape[0]}")
    print(f"  Grid: {grid}x{grid}={G2}  |  clip_max: {clip_max:.6f}")

    print(f"Optuna: {N_TRIALS} trials x {TUNE_ITERS} iters...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(
        make_objective(
            grid, cyt_idx, pde_fn, geomtime,
            X_ic, Y_ic_obs,
            X_obs_tr, Y_obs_tr,
            Y_tgt,                # scaled targets for validation
            val_indices=val_indices,
            bc_fn=bc_fn, seed=seed,
        ),
        n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
    )
    best = study.best_params
    print(f"  Best: {best}  |  val_loss = {study.best_value:.6f}")

    tf.keras.backend.clear_session(); set_seed(seed)

    WARMUP_ITERS = 5_000
    print(f"Phase 1: Data-only warmup ({WARMUP_ITERS} Adam, lambda_pde=0)...")
    model = build_dde_model(
        grid, best["hidden"], best["n_layers"], best["learning_rate"],
        pde_fn, geomtime,
        X_ic, Y_ic_obs,
        X_obs_tr, Y_obs_tr,
        bc_fn,
        lambda_pde=0.0,
        lambda_ic=best["lambda_ic"],
        lambda_bc=best["lambda_bc"],
    )
    model.train(iterations=WARMUP_ITERS, display_every=1000)

    PHYSICS_ITERS = FULL_ITERS - WARMUP_ITERS
    print(f"Phase 2: Physics-informed ({PHYSICS_ITERS} Adam, lambda_pde={best['lambda_pde']:.4f})...")
    model.compile("adam", lr=best["learning_rate"],
                   loss_weights=[best["lambda_pde"], best["lambda_ic"],
                                 best["lambda_bc"], 1.0])
    ckpt_path = str(out_dir / f"ckpt_{cytokine}_{grid}_{seed}")
    checker = dde.callbacks.ModelCheckpoint(
        ckpt_path, save_better_only=True, period=100,
    )
    model.train(iterations=PHYSICS_ITERS, display_every=2000, callbacks=[checker])

    print(f"Phase 3: L-BFGS polish ({LBFGS_ITERS} iters)...")
    dde.optimizers.set_LBFGS_options(maxiter=LBFGS_ITERS)
    model.compile("L-BFGS-B",
                   loss_weights=[best["lambda_pde"], best["lambda_ic"],
                                 best["lambda_bc"], 1.0])
    model.train(display_every=1000, callbacks=[checker])

    best_step = model.train_state.best_step
    import glob
    ckpt_files = sorted(glob.glob(ckpt_path + "*"))
    print(f"  Best step: {best_step}  |  Checkpoint files: {ckpt_files[:8]}")
    restore_path = None
    for suffix in [f"-{best_step}.ckpt", f"-{best_step}", ""]:
        candidate = ckpt_path + suffix
        if any(f.startswith(candidate) and (".index" in f or ".data-" in f) for f in ckpt_files):
            restore_path = candidate
            break
    if restore_path:
        print(f"  Restoring from: {restore_path}")
        model.restore(restore_path, verbose=1)
    else:
        print(f"  WARNING: No checkpoint found for step {best_step}, using final weights.")

    print("Predicting full grid...")
    Yp_all = np.zeros((N, grid, grid, 1), dtype=np.float64)
    for i in range(N):
        abs_t = i + 2
        tt  = np.full((G2, 1), t_to_norm(abs_t), np.float64)
        X_q = np.hstack([xy_grid, tt])
        Yp_all[i, :, :, 0] = model.predict(X_q).reshape(grid, grid)

    Y_phys  = denormalize(Y_tgt[..., cyt_idx:cyt_idx + 1], clip_max)
    Yp_phys = denormalize(Yp_all, clip_max)

    suffix = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params":          best,
        "optuna_best_val_loss": float(study.best_value),
        "results": {
            "Near_Horizon_t82_t91": calculate_metrics(
                Y_phys[80:90], Yp_phys[80:90], M_spat[80:90], clip_max),
            "Far_Horizon_t92_t100": calculate_metrics(
                Y_phys[90:99], Yp_phys[90:99], M_spat[90:99], clip_max),
        },
    }

    with open(out_dir / f"res_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    model.save(str(out_dir / f"weights_{suffix}"))
    print(f"Done! Saved: models/pinn/res_{suffix}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid",     type=int, required=True)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()
    run_pipeline(args.grid, args.seed, args.cytokine)
