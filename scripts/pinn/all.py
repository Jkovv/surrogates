import os, json, argparse, random, warnings
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

N_TRIALS        = 20
TUNE_ITERS      = 3_000
FULL_ITERS      = 20_000
N_OBS_PER_T     = 200    # observation points sampled per training timestep
N_DOMAIN        = 500    # PDE collocation points
N_BOUNDARY      = 200    # Neumann BC points
N_INITIAL       = 0      # IC enforced via PointSetBC, not sampled

# true_size=5mm, nx=grid, s_mcs=60 MCS/h, h_mcs=1/60
TRUE_SIZE = 5.0    # mm — physical domain size
S_MCS     = 60.0   # MCS per hour
H_MCS     = 1.0 / S_MCS

CYTOKINE_NAMES  = ["il8", "il1", "il6", "il10", "tnf", "tgf"]

# Cell-type index → column in Y_masks_pinn
# masks order: EC(0), NN(1), NA(2), M1(3), M2(4)
MASK_E   = 0   # EC  — endothelial
MASK_NDN = 1   # NN  — neutrophil / ndn
MASK_NA  = 2   # NA  — neutrophil activated
MASK_M1  = 3   # M1  — macrophage type 1
MASK_M2  = 4   # M2  — macrophage type 2


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    dde.config.set_random_seed(seed)

def compute_pde_params(G):
    areaconv   = TRUE_SIZE**2 / G**2
    volumeconv = (TRUE_SIZE**2 * 1.0) / (G**2 * 1.0)

    D = np.array([
        2.09e-6 * S_MCS / areaconv,   # il8
        3.00e-7 * S_MCS / areaconv,   # il1
        8.49e-8 * S_MCS / areaconv,   # il6
        1.45e-8 * S_MCS / areaconv,   # il10
        4.07e-9 * S_MCS / areaconv,   # tnf
        2.60e-7 * S_MCS / areaconv,   # tgf
    ], dtype=np.float64)

    k = np.array([
        0.200 * H_MCS,          # il8
        0.600 * H_MCS,          # il1
        0.500 * H_MCS,          # il6
        0.500 * H_MCS,          # il10
        0.500 * 0.225 * H_MCS,  # tnf
        0.500 / 25.0  * H_MCS,  # tgf
    ], dtype=np.float64)

    # Secretion rate constants
    sec = np.array([
        234e-5  * volumeconv * H_MCS,  # keil8   (s1[0]: EC)
        1.46e-5 * volumeconv * H_MCS,  # kndnil8 (s2[0]: NN/ndn)
        3.024e-5* volumeconv * H_MCS,  # thetanail8 endocytosis (e[0]: NA)
        225e-5  * volumeconv * H_MCS,  # knail1  (s1[1]: NA)
        250e-5  * volumeconv * H_MCS,  # km1il6  (s1[2]: M1)
        45e-5   * volumeconv * H_MCS,  # km2il10 (s1[3]: M1 → note: il10 from M1 in original)
        250e-5  * volumeconv * H_MCS,  # knatnf  (s1[4]: NA)
        70e-5   * volumeconv * H_MCS,  # km1tnf  (s2[4]: M1)
        280e-5  * volumeconv * H_MCS,  # km2tgf  (s1[5]: M2)
    ], dtype=np.float64)

    return D, k, sec


def build_source_terms(masks_flat, D, k, sec, G):
    n = G * G
    me  = masks_flat[:, MASK_E]    # (G*G,)
    mnn = masks_flat[:, MASK_NDN]
    mna = masks_flat[:, MASK_NA]
    mm1 = masks_flat[:, MASK_M1]
    mm2 = masks_flat[:, MASK_M2]

    zeros = np.zeros(n, dtype=np.float64)

    s1 = np.stack([
        sec[0] * me,     # il8:  EC secretion
        sec[3] * mna,    # il1:  NA secretion
        sec[4] * mm1,    # il6:  M1 secretion
        sec[5] * mm1,    # il10: M1 secretion (km2il10 uses M1 in Ioannis)
        sec[6] * mna,    # tnf:  NA secretion
        sec[8] * mm2,    # tgf:  M2 secretion
    ], axis=-1)          # (G*G, 6)

    s2 = np.stack([
        sec[1] * mnn,    # il8:  NN secondary secretion
        zeros, zeros, zeros,
        sec[7] * mm1,    # tnf:  M1 secondary secretion
        zeros,
    ], axis=-1)          # (G*G, 6)

    e = np.stack([
        sec[2] * mna,    # il8:  NA endocytosis
        zeros, zeros, zeros, zeros, zeros,
    ], axis=-1)          # (G*G, 6)

    return (tf.constant(s1, dtype=tf.float64),
            tf.constant(s2, dtype=tf.float64),
            tf.constant(e,  dtype=tf.float64))

def make_pde(D_tf, k_tf, s1_tf, s2_tf, e_tf, G):
    G_tf = tf.constant(G, dtype=tf.float64)

    def pde(x, y):
        ftype = tf.float64
        G_dyn = tf.cast(G_tf, ftype)

        lap = []
        for i in range(6):
            ui   = y[:, i:i+1]
            d2x  = dde.grad.hessian(ui, x, i=0, j=0)
            d2y  = dde.grad.hessian(ui, x, i=1, j=1)
            lap.append(d2x + d2y)
        laplacian_u = tf.concat(lap, axis=1)   # (N, 6)

        ut = tf.concat([
            dde.grad.jacobian(y[:, i:i+1], x, i=0, j=2)
            for i in range(6)
        ], axis=1)                             # (N, 6)

        ix = tf.cast(tf.clip_by_value(
            tf.floor((x[:, 0:1] + tf.cast(1.0, ftype)) / tf.cast(2.0, ftype) * G_dyn),
            tf.cast(0.0, ftype), G_dyn - tf.cast(1.0, ftype)
        ), tf.int32)                           # (N, 1)
        iy = tf.cast(tf.clip_by_value(
            tf.floor((x[:, 1:2] + tf.cast(1.0, ftype)) / tf.cast(2.0, ftype) * G_dyn),
            tf.cast(0.0, ftype), G_dyn - tf.cast(1.0, ftype)
        ), tf.int32)                           # (N, 1)
        flat_idx = tf.squeeze(ix * tf.cast(G_tf, tf.int32) + iy, axis=1)  # (N,)

        s1_q = tf.cast(tf.gather(s1_tf, flat_idx), ftype)
        s2_q = tf.cast(tf.gather(s2_tf, flat_idx), ftype)
        e_q  = tf.cast(tf.gather(e_tf,  flat_idx), ftype)
        D_dyn = tf.cast(D_tf, ftype)
        k_dyn = tf.cast(k_tf, ftype)

        # res
        degradation = k_dyn * y
        endocytosis = e_q * y
        rhs = D_dyn * laplacian_u - degradation + s1_q + s2_q - endocytosis
        return ut - rhs                        # (N, 6) — should be 0

    return pde


# metrics 
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

def build_dde_model(G, hidden, n_layers, lr,
                    pde_fn, geomtime,
                    X_ic, Y_ic_obs,
                    X_obs, Y_obs,
                    bc_fn,
                    lambda_pde=1.0, lambda_ic=10.0, lambda_bc=1.0):
    # IC constraints 
    ic_bcs = [
        dde.PointSetBC(X_ic, Y_ic_obs[:, i:i+1], component=i)
        for i in range(6)
    ]

    # Neumann zero-flux BCs
    bcs = [
        dde.NeumannBC(geomtime, lambda x: np.zeros((len(x), 1), np.float64),
                      bc_fn, component=i)
        for i in range(6)
    ]

    # Observation constraints (training data)
    obs_bcs = [
        dde.PointSetBC(X_obs, Y_obs[:, i:i+1], component=i)
        for i in range(6)
    ]

    data = dde.data.TimePDE(
        geomtime, pde_fn,
        ic_bcs=ic_bcs + bcs + obs_bcs,
        num_domain=N_DOMAIN,
        num_boundary=N_BOUNDARY,
        num_initial=N_INITIAL,
        train_distribution="uniform",
    )

    net = dde.maps.FNN(
        [3] + [hidden] * n_layers + [6],
        "tanh", "Glorot uniform"
    )
    net.apply_output_transform(lambda x, y: tf.nn.softplus(y) - 1.0)
    # softplus(y)-1 → output ≈ 0 at y=0, allows negative values near 0
    # (cytokines scaled to [-1,1] so negative is valid)

    # Loss weights: [pde, bc×6, ic×6, obs×6]
    n_pde = 1
    n_bc  = 6
    n_ic  = 6
    n_obs = 6
    loss_weights = (
        [lambda_pde] * n_pde +
        [lambda_bc]  * n_bc  +
        [lambda_ic]  * n_ic  +
        [1.0]        * n_obs
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    return model


# Optuna 
def make_objective(G, pde_fn, geomtime, X_ic, Y_ic_obs,
                   X_obs_tr, Y_obs_tr, X_obs_vl, Y_obs_vl,
                   arrays_phys, val_times, bc_fn, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()

        hidden     = trial.suggest_categorical("hidden",      [32, 64, 128])
        n_layers   = trial.suggest_categorical("n_layers",    [3, 4, 5])
        lr         = trial.suggest_float("learning_rate",     1e-4, 5e-3, log=True)
        lambda_pde = trial.suggest_float("lambda_pde",        0.01, 10.0, log=True)
        lambda_ic  = trial.suggest_float("lambda_ic",         1.0,  100.0, log=True)

        model = build_dde_model(
            G, hidden, n_layers, lr,
            pde_fn, geomtime,
            X_ic, Y_ic_obs,
            X_obs_tr, Y_obs_tr,
            bc_fn,
            lambda_pde=lambda_pde, lambda_ic=lambda_ic,
        )
        model.train(iterations=TUNE_ITERS, display_every=0)

        # Validation MSE over all val timesteps
        xy_grid = make_xy_grid(G)
        mses = []
        for t in val_times:
            tt   = np.full((xy_grid.shape[0], 1), t_to_norm(t, G), np.float64)
            X_q  = np.hstack([xy_grid, tt])
            Yp   = model.predict(X_q)                    # (G*G, 6)
            Yt   = arrays_phys[t].reshape(-1, 6).astype(np.float64)
            mses.append(float(np.mean((Yt - Yp)**2)))
        return float(np.mean(mses))
    return objective

def make_xy_grid(G):
    xs = np.linspace(-1.0, 1.0, G, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, G, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)  # (G*G, 2)

def t_to_norm(t_idx, n_total=101, window=2):
    t_norm_all = np.linspace(-1.0, 1.0, n_total, dtype=np.float64)
    return float(t_norm_all[t_idx + window])


def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    cyt_idx = CYTOKINE_NAMES.index(cytokine.lower())

    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir   = Path("./models/pinn"); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}×{grid} — loading data...")

    Y_raw   = np.load(data_path/"Y_raw_phys.npy").astype(np.float64)   # (101,G,G,6)
    Y_tgt   = np.load(data_path/"Y_target.npy").astype(np.float64)     # (99,G,G,6)
    M_spat  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float64) # (99,G,G,5)
    M_pinn  = np.load(data_path/"Y_masks_pinn.npy").astype(np.float64)  # (99,G*G,5)
    Y_ic_raw= np.load(data_path/"Y_ic.npy").astype(np.float64)          # (G,G,6) scaled
    X_col   = np.load(data_path/"X_colloc.npy").astype(np.float64)      # (99,G*G,3)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_maxs = np.array(meta["scaling"]["max"], dtype=np.float64)
    clip_max  = clip_maxs[cyt_idx]

    G  = grid
    N  = Y_tgt.shape[0]    # 99
    G2 = G * G

    # PDE setup 
    D_arr, k_arr, sec_arr = compute_pde_params(G)
    # mean mask over training timesteps for steady spatial source terms
    masks_mean = M_pinn[:70].mean(axis=0)  
    s1_tf, s2_tf, e_tf = build_source_terms(masks_mean, D_arr, k_arr, sec_arr, G)

    D_tf = tf.constant(D_arr.reshape(1, 6), dtype=tf.float64)
    k_tf = tf.constant(k_arr.reshape(1, 6), dtype=tf.float64)

    xy_grid    = make_xy_grid(G)
    xy_grid_tf = tf.constant(xy_grid, dtype=tf.float64)

    pde_fn = make_pde(D_tf, k_tf, s1_tf, s2_tf, e_tf, G)

    t_norm_all = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    t_min = float(t_norm_all[2])   
    t_max = float(t_norm_all[-1])

    geom      = dde.geometry.Rectangle([-1.0, -1.0], [1.0, 1.0])
    timedomain= dde.geometry.TimeDomain(t_min, t_max)
    geomtime  = dde.geometry.GeometryXTime(geom, timedomain)

    def bc_fn(x, on_boundary):
        return on_boundary

    # IC constraints 
    tt0 = np.full((G2, 1), t_min, dtype=np.float64)
    X_ic     = np.hstack([xy_grid, tt0])                     # (G*G, 3)
    Y_ic_obs = Y_ic_raw.reshape(G2, 6).astype(np.float64)    # (G*G, 6)

    # obs constraints 
    rng = np.random.default_rng(seed)
    train_indices = list(range(70))     # Y_tgt index 0..69
    val_indices   = list(range(70, 80))

    X_obs_list_tr = []; Y_obs_list_tr = []
    for i in train_indices:
        pts_idx = rng.choice(G2, size=min(N_OBS_PER_T, G2), replace=False)
        xy_obs  = xy_grid[pts_idx]
        t_obs   = np.full((len(pts_idx), 1), t_norm_all[i + 2], np.float64)
        X_obs_list_tr.append(np.hstack([xy_obs, t_obs]))
        Y_obs_list_tr.append(Y_tgt[i].reshape(G2, 6)[pts_idx])

    X_obs_tr = np.vstack(X_obs_list_tr).astype(np.float64)  # (80*N_OBS, 3)
    Y_obs_tr = np.vstack(Y_obs_list_tr).astype(np.float64)  # (80*N_OBS, 6)

    X_obs_list_vl = []; Y_obs_list_vl = []
    for i in val_indices:
        pts_idx = rng.choice(G2, size=min(N_OBS_PER_T, G2), replace=False)
        xy_obs  = xy_grid[pts_idx]
        t_obs   = np.full((len(pts_idx), 1), t_norm_all[i + 2], np.float64)
        X_obs_list_vl.append(np.hstack([xy_obs, t_obs]))
        Y_obs_list_vl.append(Y_tgt[i].reshape(G2, 6)[pts_idx])

    X_obs_vl = np.vstack(X_obs_list_vl).astype(np.float64)
    Y_obs_vl = np.vstack(Y_obs_list_vl).astype(np.float64)

    print(f"  Train obs: {X_obs_tr.shape[0]}  |  Val obs: {X_obs_vl.shape[0]}")
    print(f"  IC points: {X_ic.shape[0]}  |  Grid: {G}×{G}={G2}")

    # optuna 
    print(f"Optuna: {N_TRIALS} trials × {TUNE_ITERS} iters...")
    arrays_phys = Y_raw  

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(
        make_objective(G, pde_fn, geomtime,
                       X_ic, Y_ic_obs,
                       X_obs_tr, Y_obs_tr,
                       X_obs_vl, Y_obs_vl,
                       arrays_phys,
                       val_times=[i + 2 for i in val_indices], 
                       bc_fn=bc_fn,
                       seed=seed),
        n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
    )
    best = study.best_params
    print(f"  Best: {best}  |  val_loss = {study.best_value:.6f}")

    # final training 
    tf.keras.backend.clear_session(); set_seed(seed)
    model = build_dde_model(
        G, best["hidden"], best["n_layers"], best["learning_rate"],
        pde_fn, geomtime,
        X_ic, Y_ic_obs,
        X_obs_tr, Y_obs_tr,
        bc_fn,
        lambda_pde=best["lambda_pde"],
        lambda_ic=best["lambda_ic"],
    )
    print(f"Final training [{cytokine.upper()}] {grid}×{grid}  ({FULL_ITERS} iters)...")
    model.train(iterations=FULL_ITERS, display_every=2000)

    print("Predicting full grid...")
    Yp_all = np.zeros((N, G, G, 6), dtype=np.float64)
    for i in range(N):
        tt = np.full((G2, 1), t_norm_all[i + 2], np.float64)
        X_q = np.hstack([xy_grid, tt])
        Yp_all[i] = model.predict(X_q).reshape(G, G, 6)

    Y_phys  = denormalize(Y_tgt[..., cyt_idx:cyt_idx+1], clip_max)   # (99,G,G,1)
    Yp_phys = denormalize(Yp_all[..., cyt_idx:cyt_idx+1], clip_max)  # (99,G,G,1)

    # eval
    suffix  = f"{cytokine}_{grid}_{seed}"
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
