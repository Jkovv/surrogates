import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
import optuna
import gc
import time
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import warnings

# for now 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
warnings.filterwarnings("ignore")

import deepxde as dde
dde.config.disable_xla_jit() # for now 

CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    dde.config.set_random_seed(seed)

def get_pde_params():
    D = np.array([2.09e-6, 3e-7, 8.49e-8, 1.45e-8, 4.07e-9, 2.6e-7])
    K = np.array([0.2, 0.6, 0.5, 0.5, 0.11, 0.02])
    return tf.constant(D, dtype=tf.float32), tf.constant(K, dtype=tf.float32)

def wound_pde(x, y, D_tf, k_tf):
    u_list = [y[:, i:i+1] for i in range(6)]
    res = []
    for i in range(6):
        du_t = dde.grad.jacobian(y, x, i=i, j=2)
        du_xx = dde.grad.hessian(y, x, component=i, i=0, j=0)
        du_yy = dde.grad.hessian(y, x, component=i, i=1, j=1)
        res.append(du_t - (D_tf[i] * (du_xx + du_yy) - k_tf[i] * u_list[i]))
    return tf.concat(res, axis=1)

def calculate_metrics_masked(y_true, y_pred, masks, grid):
    yt, yp = y_true.flatten(), y_pred.flatten()
    mask = np.max(masks, axis=-1).flatten()
    
    yt_m, yp_m = yt[mask > 0], yp[mask > 0]
    if len(yt_m) == 0: 
        return {"Global_R2": 0.0, "Masked_RMSE": 0.0, "Avg_SSIM": 0.0}

    r2 = r2_score(yt_m, yp_m)
    rmse = np.sqrt(np.mean(np.square(yt_m - yp_m)))
    
    y_t_sq = y_true.reshape(-1, grid, grid)
    y_p_sq = y_pred.reshape(-1, grid, grid)
    ssim_v = np.mean([ssim(y_t_sq[i], y_p_sq[i], data_range=max(y_t_sq[i].max(), 1.0)) for i in range(len(y_t_sq))])
    
    return {"Global_R2": float(r2), "Masked_RMSE": float(rmse), "Avg_SSIM": float(ssim_v)}

def build_pinn_model(grid, cytokine, lr, D_tf, k_tf, X_train, Y_train):
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    obs = dde.icbc.PointSetBC(X_train, Y_train, component=CYTOKINE_MAP[cytokine])

    data = dde.data.TimePDE(
        geomtime, 
        lambda x, y: wound_pde(x, y, D_tf, k_tf),
        [obs],
        num_domain=5000, 
        num_boundary=1000,
        anchors=X_train
    )

    net = dde.maps.FNN([3] + [128] * 4 + [6], "tanh", "Glorot uniform")
    net.apply_output_transform(lambda x, y: tf.nn.softplus(y))
    
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    return model

def run_pinn_full():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cyto = args.cytokine.lower()
    data_path = Path(f"./preprocessed/{args.grid}x{args.grid}")
    out_dir = Path("./models/pinn")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    X_trunk = np.load(data_path / "X_trunk.npy").astype(np.float32)
    Y_target = np.load(data_path / "Y_target.npy").astype(np.float32).reshape(-1, 6)
    M_all = np.load(data_path / "Y_masks.npy").astype(np.float32)
    
    num_train_full = int(0.7 * len(X_trunk))
    X_train_pts = X_trunk[:num_train_full].reshape(-1, 3)
    Y_train_pts = Y_target[:num_train_full*args.grid*args.grid]

    D_tf, k_tf = get_pde_params()

    print(f"\nHyperparameter Search (5k points, Seed: {args.seed})...")
    idx_optuna = np.random.choice(len(X_train_pts), 5000, replace=False)
    X_opt, Y_opt = X_train_pts[idx_optuna], Y_train_pts[idx_optuna]

    def objective(trial):
        tf.keras.backend.clear_session()
        gc.collect()
        trial_lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        m = build_pinn_model(args.grid, cyto, trial_lr, D_tf, k_tf, X_opt, Y_opt)
        res = m.train(iterations=1000, display_every=1000) 
        return res[1].best_loss_train

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_lr = study.best_params['lr']

    tf.keras.backend.clear_session()
    gc.collect()
    time.sleep(2)

    print(f"Starting Training for {cyto.upper()}...")
    idx_final = np.random.choice(len(X_train_pts), min(50000, len(X_train_pts)), replace=False)
    X_final, Y_final = X_train_pts[idx_final], Y_train_pts[idx_final]

    final_model = build_pinn_model(args.grid, cyto, best_lr, D_tf, k_tf, X_final, Y_final)
    final_model.train(iterations=12000, display_every=1000)

    Y_pred_all = np.array([final_model.predict(X_trunk[t]) for t in range(len(X_trunk))])
    
    with open(data_path / "scaling_params.json", 'r') as f:
        scales = json.load(f)
    
    c_idx = CYTOKINE_MAP[cyto]
    Y_pred_phys = np.expm1(Y_pred_all[..., c_idx:c_idx+1] * scales[cyto])
    Y_true_phys = np.expm1(Y_target.reshape(len(X_trunk), args.grid, args.grid, 6)[..., c_idx:c_idx+1] * scales[cyto])

    res_json = {
        "params": study.best_params, "seed": args.seed, "grid": args.grid, "cytokine": cyto,
        "results": {
            "Interpolation_72_89": calculate_metrics_masked(Y_true_phys[70:88], Y_pred_phys[70:88], M_all[70:88], args.grid),
            "Extrapolation_82_100": calculate_metrics_masked(Y_true_phys[80:99], Y_pred_phys[80:99], M_all[80:99], args.grid)
        }
    }

    suffix = f"results_pinn_{cyto}_{args.grid}_s{args.seed}"
    with open(out_dir / f"{suffix}.json", 'w') as f:
        json.dump(res_json, f, indent=4)
    
    final_model.save(str(out_dir / f"weights_{suffix}"))
    print(f"\nSaved results and weights for {suffix}")

if __name__ == "__main__":
    run_pinn_full()
