import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
import deepxde as dde
import optuna
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore")
tf.keras.backend.set_floatx('float32')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# metrics
def compute_dice_coefficient(y_true, y_pred, smooth=1e-6):
    threshold = 0.05 
    y_true_bin = (y_true > threshold).astype(np.float32)
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin)
    return (2. * intersection + smooth) / (union + smooth)

def calculate_metrics(y_true, y_pred, masks):
    if y_true.ndim == 4: y_true = y_true[..., 0]
    if y_pred.ndim == 4: y_pred = y_pred[..., 0]
    
    # global R2
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    # Masked RMSE
    if masks is not None:
        spatial_mask = np.max(masks, axis=-1).squeeze()
        if spatial_mask.ndim == 3: spatial_mask = spatial_mask
        sq_diff = np.square(y_true - y_pred) * spatial_mask
        m_rmse = np.sqrt(np.sum(sq_diff) / (np.sum(spatial_mask) + 1e-7))
    else:
        m_rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))

    dices, corrs, ssim_vals = [], [], []
    
    for t in range(y_true.shape[0]):
        gt, pr = y_true[t], y_pred[t]
        
        # Dice
        dices.append(compute_dice_coefficient(gt, pr))
        
        # SSIM
        drange = max(gt.max(), 1.0)
        win_size = min(7, gt.shape[0], gt.shape[1])
        if win_size % 2 == 0: win_size -= 1
        ssim_vals.append(ssim(gt, pr, data_range=drange, win_size=win_size))
        
        # Pearson
        if np.std(gt) > 1e-9 and np.std(pr) > 1e-9:
            corrs.append(pearsonr(gt.flatten(), pr.flatten())[0])
            
    return {
        "Global_R2": float(r2), 
        "Masked_RMSE": float(m_rmse),
        "Avg_Dice": float(np.mean(dices)),
        "Avg_SSIM": float(np.mean(ssim_vals)),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0
    }

def get_pde_constants(grid_size):
    nx = grid_size
    true_size = 5
    s_mcs = 60.0
    h_mcs = 1 / 60.0
    areaconv = true_size**2 / nx**2
    volumeconv = (true_size**2 * 1) / (nx**2 * 1)

    vals = {
        'Dil8': 2.09e-6 * s_mcs / areaconv, 'muil8': 0.2 * h_mcs,
        'keil8': 234e-5 * volumeconv * h_mcs, 'kndnil8': 1.46e-5 * volumeconv * h_mcs, 'thetanail8': 3.024e-5 * volumeconv * h_mcs,
        'Dil1': 3e-7 * s_mcs / areaconv, 'muil1': 0.6 * h_mcs, 'knail1': 225e-5 * volumeconv * h_mcs,
        'Dil6': 8.49e-8 * s_mcs / areaconv, 'muil6': 0.5 * h_mcs, 'km1il6': 250e-5 * volumeconv * h_mcs,
        'Dil10': 1.45e-8 * s_mcs / areaconv, 'muil10': 0.5 * h_mcs, 'km2il10': 45e-5 * volumeconv * h_mcs,
        'Dtnf': 4.07e-9 * s_mcs / areaconv, 'mutnf': 0.5 * 0.225 * h_mcs, 'knatnf': 250e-5 * volumeconv * h_mcs, 'km1tnf': 70e-5 * volumeconv * h_mcs,
        'Dtgf': 2.6e-7 * s_mcs / areaconv, 'mutgf': 0.5 * (1 / 25) * h_mcs, 'km2tgf': 280e-5 * volumeconv * h_mcs
    }
    return vals

def get_source_terms(T_MAX):
    cp_e = np.array([1]*90 + [1]) 
    cp_ndn = np.array([1]*10 + [0]*10 + [1]*5 + [0]*66) 
    cp_na = np.array([1]*89 + [0, 0])
    cp_m1 = np.array([1]*89 + [0, 0])
    cp_m2 = np.array([1]*91)

    def pad(arr, length):
        if len(arr) >= length: return arr[:length]
        return np.pad(arr, (0, length - len(arr)), 'edge')

    return (pad(x, T_MAX+1) for x in [cp_e, cp_ndn, cp_na, cp_m1, cp_m2])

class AdaptiveLossWeightsCallback(dde.callbacks.Callback):
    """Normalized Adaptive Weights (NAW) - Balances Physics vs Data"""
    def __init__(self, update_every=500, alpha=0.9):
        super().__init__()
        self.update_every = update_every
        self.alpha = alpha
        self.iter = 0

    def on_train_begin(self):
        if not hasattr(self.model, 'loss_weights') or self.model.loss_weights is None:
             n_losses = len(self.model.train_state.loss_train)
             self.model.loss_weights = np.ones(n_losses)

    def on_epoch_end(self):
        self.iter += 1
        if self.iter % self.update_every == 0:
            losses = np.array(self.model.train_state.loss_train)
            mean_loss = np.mean(losses)
            new_weights = mean_loss / (losses + 1e-8)
            
            if hasattr(self.model, 'loss_weights') and self.model.loss_weights is not None:
                old_weights = self.model.loss_weights
                self.model.loss_weights = self.alpha * old_weights + (1 - self.alpha) * new_weights

def run_pinn(grid, seed, cytokine):
    set_seed(seed)
    
    base_dir = Path(".")
    data_path = base_dir / f"preprocessed/{grid}x{grid}"
    out_dir = base_dir / "models/pinn_optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Data: {cytokine} | Grid {grid}...")
    try:
        Y_all = np.load(data_path / "Y_target.npy").astype(np.float32) 
        M_all = np.load(data_path / "Y_masks.npy").astype(np.float32)
    except FileNotFoundError:
        print(f"Data missing at {data_path}")
        return

    idx = CYTOKINE_MAP[cytokine]
    Y_target = Y_all[..., idx:idx+1]
    T_MAX = Y_target.shape[0] - 1
    
    pc = get_pde_constants(grid)
    D_vec = np.array([pc['Dil8'], pc['Dil1'], pc['Dil6'], pc['Dil10'], pc['Dtnf'], pc['Dtgf']])
    k_vec = np.array([pc['muil8'], pc['muil1'], pc['muil6'], pc['muil10'], pc['mutnf'], pc['mutgf']])
    
    D_tf = tf.constant(D_vec.reshape(1, 6), dtype=tf.float32)
    k_tf = tf.constant(k_vec.reshape(1, 6), dtype=tf.float32)
    
    cp_e, cp_ndn, cp_na, cp_m1, cp_m2 = get_source_terms(T_MAX)
    
    s1_np = np.stack([
        pc['keil8'] * cp_e, pc['knail1'] * cp_na, pc['km1il6'] * cp_m1,
        pc['km2il10'] * cp_m1, pc['knatnf'] * cp_na, pc['km2tgf'] * cp_m2
    ])
    s2_np = np.stack([
        pc['kndnil8'] * cp_ndn, np.zeros_like(cp_ndn), np.zeros_like(cp_ndn),
        np.zeros_like(cp_ndn), pc['km1tnf'] * cp_m1, np.zeros_like(cp_ndn)
    ])
    e_np = np.stack([
        pc['thetanail8'] * cp_na, np.zeros_like(cp_ndn), np.zeros_like(cp_ndn),
        np.zeros_like(cp_ndn), np.zeros_like(cp_ndn), np.zeros_like(cp_ndn)
    ])
    
    s1_tf = tf.constant(s1_np, dtype=tf.float32)
    s2_tf = tf.constant(s2_np, dtype=tf.float32)
    e_tf  = tf.constant(e_np, dtype=tf.float32)

    def pde(x, y):
        # x: (N, 3) [x, y, t]
        u = [y[:, i:i+1] for i in range(6)]
        
        lap = []
        for i in range(6):
            d2x = dde.grad.hessian(u[i], x, i=0, j=0)
            d2y = dde.grad.hessian(u[i], x, i=1, j=1)
            lap.append(d2x + d2y)
        laplacian_u = tf.concat(lap, axis=1)
        
        ut = []
        for i in range(6): ut.append(dde.grad.jacobian(u[i], x, i=0, j=2))
        u_t = tf.concat(ut, axis=1)
        
        t = x[:, 2]
        time_idx = tf.cast(tf.round(t), tf.int32)
        time_idx = tf.clip_by_value(time_idx, 0, T_MAX)
        
        s1_curr = tf.gather(tf.transpose(s1_tf), time_idx)
        s2_curr = tf.gather(tf.transpose(s2_tf), time_idx)
        e_curr  = tf.gather(tf.transpose(e_tf), time_idx)
        
        rhs = laplacian_u * D_tf - k_tf * y + s1_curr + s2_curr - e_curr * y
        return u_t - rhs

    geom = dde.geometry.Rectangle([0, 0], [grid-1, grid-1])
    timedomain = dde.geometry.TimeDomain(0, T_MAX)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    coords = np.stack(np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij"), axis=-1).reshape(-1, 2)
    
    obs_X, obs_Y = [], []
    train_time_limit = int(0.7 * T_MAX)
    
    for t in range(train_time_limit):
        n_samples = max(50, int(0.05 * len(coords))) # 5% sampling
        idx_rand = np.random.choice(len(coords), n_samples, replace=False)
        pts = coords[idx_rand]
        obs_X.append(np.hstack([pts, np.full((n_samples, 1), t)]))
        obs_Y.append(Y_all[t].reshape(-1, 6)[idx_rand])
        
    X_train = np.vstack(obs_X).astype(np.float32)
    Y_train = np.vstack(obs_Y).astype(np.float32)
    
    # BC
    bcs = [dde.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary, component=i) for i in range(6)]
    data_bcs = [dde.PointSetBC(X_train, Y_train[:, i:i+1], component=i) for i in range(6)]
    
    # IC (t=0)
    X_ic = np.hstack([coords, np.zeros((len(coords), 1))]).astype(np.float32)
    Y_ic = Y_all[0].reshape(-1, 6).astype(np.float32)
    ic_bcs = [dde.PointSetBC(X_ic, Y_ic[:, i:i+1], component=i) for i in range(6)]

    full_bcs = bcs + data_bcs + ic_bcs

    # optuna
    def objective(trial):
        tf.keras.backend.clear_session()
        
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        depth = trial.suggest_int("layers", 3, 6)
        width = trial.suggest_categorical("neurons", [32, 64, 128])
        activation = "tanh"
        
        data = dde.data.TimePDE(
            geomtime, pde, ic_bcs=full_bcs,
            num_domain=1000, num_boundary=200, num_initial=0
        )
        
        net = dde.maps.FNN([3] + [width] * depth + [6], activation, "Glorot uniform")
        net.apply_output_transform(lambda x, y: tf.nn.softplus(y))
        
        model = dde.Model(data, net)
        model.compile("adam", lr=lr)
        
        losshistory, _ = model.train(iterations=2000, display_every=2000)
        
        return losshistory[-1] 

    print("Optuna Search...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5) # 5 trials for speed
    best = study.best_params
    print(f"Best PINN Params: {best}")

    print("starting the run (15k epochs)...")
    tf.keras.backend.clear_session()
    
    data = dde.data.TimePDE(
        geomtime, pde, ic_bcs=full_bcs,
        num_domain=2500, num_boundary=500, num_initial=0
    )
    
    net = dde.maps.FNN([3] + [best['neurons']] * best['layers'] + [6], "tanh", "Glorot uniform")
    net.apply_output_transform(lambda x, y: tf.nn.softplus(y))
    
    model = dde.Model(data, net)
    model.compile("adam", lr=best['lr'])
    
    suffix = f"{cytokine}_grid{grid}_seed{seed}_BEST_lr{best['lr']:.1e}_L{best['layers']}_N{best['neurons']}"
    
    # NAW + checkpoint
    naw_cb = AdaptiveLossWeightsCallback(update_every=500)
    ckpt_cb = dde.callbacks.ModelCheckpoint(
        str(out_dir / f"model_{suffix}.ckpt"), save_better_only=True, period=1000
    )
    
    model.train(iterations=15000, display_every=1000, callbacks=[naw_cb, ckpt_cb])

    # evaluation 
    print("Evaluating...")
    
    def predict_grid(t_step):
        g_coords = np.stack(np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij"), axis=-1).reshape(-1, 2)
        X_eval = np.hstack([g_coords, np.full((len(g_coords), 1), t_step)]).astype(np.float32)
        return model.predict(X_eval).reshape(grid, grid, 6)

    Y_pred_full = []
    for t in range(T_MAX + 1):
        Y_pred_full.append(predict_grid(t))
    Y_pred_full = np.array(Y_pred_full)

    Y_pred_target = Y_pred_full[..., idx]
    Y_true_target = Y_target[..., 0]

    res = {
        "params": best, "seed": seed, "grid": grid, "cytokine": cytokine,
        "results": {
            "Interpolation_72_89": calculate_metrics(Y_true_target[72:89], Y_pred_target[72:89], M_all[72:89]),
            "Extrapolation_82_100": calculate_metrics(Y_true_target[82:100], Y_pred_target[82:100], M_all[82:100])
        }
    }

    with open(out_dir / f"results_{suffix}.json", 'w') as f:
        json.dump(res, f, indent=4)
        
    net.save_weights(str(out_dir / f"weights_{suffix}.weights.h5"))
    print(f"Saved to results_{suffix}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pinn(args.grid, args.seed, args.cytokine.lower())
