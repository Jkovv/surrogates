import os, json, argparse, random
from pathlib import Path

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS    = 20
TUNE_EPOCHS = 30
FULL_EPOCHS = 400
N_SENSORS   = 8192   # sensor points sampled per step
EVAL_CHUNK  = 2048   # trunk chunk size at evaluation

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

#trunk 
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
        b = self.branch(xb, training=training)           # (batch, p)
        t = self.trunk(xt)                               # (batch, n_pts, p)
        r = tf.einsum("bp,bnp->bn", b, t) + self.bias   # (batch, n_pts)
        return tf.expand_dims(r, -1)                     # (batch, n_pts, 1)

def prepare_sensor_data(Xb, Xt, Yf):
    N, _, G, _, C = Xb.shape
    Xs = Xb.transpose(0, 2, 3, 1, 4).reshape(N, G * G, 22).astype(np.float32)
    return Xs, Xt, Yf

def build_dataset(Xs, Xt, Yf, batch_size, m, shuffle=True):
    N, n_pts, n_ch = Xs.shape
    actual = min(n_pts, m)

    def gen():
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)
        for i in idx:
            pi = np.random.choice(n_pts, size=actual, replace=False)
            yield (Xs[i, pi].reshape(-1), Xt[i, pi]), Yf[i, pi]

    sig = (
        (tf.TensorSpec((actual * n_ch,), tf.float32),
         tf.TensorSpec((actual, 3),      tf.float32)),
        tf.TensorSpec((actual, 1),       tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig) \
                          .batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_step(model, opt, xb, xt, y):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.square(model([xb, xt], training=True) - y))
    opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                            model.trainable_variables))
    return float(loss)

def val_step(model, xb, xt, y):
    return float(tf.reduce_mean(
        tf.square(model([xb, xt], training=False) - y)
    ))

def train_model(model, opt, ds_tr, ds_vl,
                epochs, patience=40, reduce_patience=15, min_lr=1e-7,
                verbose=True):
    best_val = np.inf; best_w = None; wait = rw = 0

    for ep in range(1, epochs + 1):
        tr = np.mean([train_step(model, opt, *b[0], b[1]) for b in ds_tr])
        vl = np.mean([val_step(model,       *b[0], b[1]) for b in ds_vl])

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
                if verbose:
                    print(f"  LR → {new_lr:.2e}")
            rw = 0

        if wait >= patience:
            if verbose: print(f"  Early stop @ epoch {ep}")
            break

    if best_w:
        model.set_weights(best_w)
    return best_val


# eval 
def predict_full(model, Xs, Xt, m=N_SENSORS, chunk=EVAL_CHUNK):
    """
    Full-grid eval: branch uses fixed m sensors, trunk chunked over all G² pts.
    """
    N, n_pts, n_ch = Xs.shape
    actual = min(n_pts, m)
    rng = np.random.default_rng(0)
    sensor_idx = rng.choice(n_pts, size=actual, replace=False)
    xb_all = tf.constant(Xs[:, sensor_idx].reshape(N, -1))  # (N, actual*n_ch)

    out = np.zeros((N, n_pts, 1), np.float32)
    for s in range(0, n_pts, chunk):
        e = min(s + chunk, n_pts)
        out[:, s:e] = model(
            [xb_all, tf.constant(Xt[:, s:e])], training=False
        ).numpy()
    return out


# metrics 
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
        dr=float(np.max(gt)-np.min(gt))
        if dr>1e-12:
            ssims_v.append(float(ssim(gt,pr,data_range=dr)))

    return {
        "Global_R2":           r2,
        "Masked_RMSE":         rmse,
        "Avg_Dice":            float(np.mean(dices)),
        "Spatial_Correlation": float(np.mean(corrs))   if corrs   else 0.0,
        "SSIM":                float(np.mean(ssims_v)) if ssims_v else 0.0,
    }

def denormalize(x, clip_max):
    return (np.asarray(x, np.float64)+1.0)/2.0*clip_max

def make_objective(Xs_tr, Xt_tr, Yf_tr, Xs_vl, Xt_vl, Yf_vl, seed):
    def objective(trial):
        set_seed(seed)
        tf.keras.backend.clear_session()
        p      = trial.suggest_categorical("p",          [64, 128, 256])
        hidden = trial.suggest_categorical("hidden",     [128, 256])
        lr     = trial.suggest_float("learning_rate",    1e-5, 1e-3, log=True)
        bs     = trial.suggest_categorical("batch_size", [4, 8])

        ds_tr = build_dataset(Xs_tr, Xt_tr, Yf_tr, bs, N_SENSORS, shuffle=True)
        ds_vl = build_dataset(Xs_vl, Xt_vl, Yf_vl, bs, N_SENSORS, shuffle=False)

        model = DeepONet(hidden=hidden, p=p)
        opt   = tf.keras.optimizers.Adam(lr)
        best  = train_model(model, opt, ds_tr, ds_vl,
                            epochs=TUNE_EPOCHS, patience=8,
                            reduce_patience=5, verbose=False)
        return float(best)
    return objective


def run_pipeline(grid, seed, cytokine):
    set_seed(seed)
    cyt_names = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
    idx = cyt_names.index(cytokine.lower())

    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir   = Path("./models/deeponet"); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{cytokine.upper()}] {grid}x{grid} — loading data...")
    Xb = np.load(data_path/"X_branch.npy").astype(np.float32)
    Xt = np.load(data_path/"X_trunk.npy").astype(np.float32)
    Y  = np.load(data_path/"Y_target.npy").astype(np.float32)[..., idx:idx+1]
    M  = np.load(data_path/"Y_masks_spatial.npy").astype(np.float32)

    with open(data_path/"metadata.json") as f:
        meta = json.load(f)
    clip_max = float(meta["scaling"]["max"][idx])

    N=Xb.shape[0]; G2=Xt.shape[1]; G=int(round(G2**0.5))
    Yf = Y.reshape(N, G2, 1)

    Xs, Xt, Yf = prepare_sensor_data(Xb, Xt, Yf)
    m_actual = min(G2, N_SENSORS)
    n_ch = Xs.shape[2]

    print(f"  Sensors: {m_actual}/{G2} pts/step  |  branch input dim: {m_actual*n_ch}")

    Xs_tr,Xt_tr,Yf_tr = Xs[:80],  Xt[:80],  Yf[:80]
    Xs_vl,Xt_vl,Yf_vl = Xs[80:90],Xt[80:90],Yf[80:90]

    #optuna
    print(f"Optuna: {N_TRIALS} trials × {TUNE_EPOCHS} epochs...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        make_objective(Xs_tr,Xt_tr,Yf_tr,Xs_vl,Xt_vl,Yf_vl,seed),
        n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,),
    )
    best = study.best_params
    print(f"  Best: {best}  |  val_loss = {study.best_value:.6f}")

    # final training
    tf.keras.backend.clear_session(); set_seed(seed)
    ds_tr = build_dataset(Xs_tr,Xt_tr,Yf_tr,best["batch_size"],m_actual,shuffle=True)
    ds_vl = build_dataset(Xs_vl,Xt_vl,Yf_vl,best["batch_size"],m_actual,shuffle=False)

    model = DeepONet(hidden=best["hidden"], p=best["p"])
    opt   = tf.keras.optimizers.Adam(best["learning_rate"])
    print(f"Final training [{cytokine.upper()}] {grid}x{grid}  (max {FULL_EPOCHS} epochs)...")
    train_model(model, opt, ds_tr, ds_vl,
                epochs=FULL_EPOCHS, patience=40, reduce_patience=15, verbose=True)

    # eval 
    Yp_flat = predict_full(model, Xs, Xt, m=m_actual)
    Yp      = Yp_flat.reshape(N, G, G, 1)
    Y_phys  = denormalize(Y.reshape(N,G,G,1), clip_max)
    Yp_phys = denormalize(Yp, clip_max)

    suffix  = f"{cytokine}_{grid}_{seed}"
    results = {
        "grid": grid, "seed": seed, "cytokine": cytokine,
        "best_params": best,
        "optuna_best_val_loss": float(study.best_value),
        "n_sensors": m_actual,
        "results": {
            "Interpolation_72_89":  calculate_metrics(
                Y_phys[70:88], Yp_phys[70:88], M[70:88]),
            "Extrapolation_82_100": calculate_metrics(
                Y_phys[80:99], Yp_phys[80:99], M[80:99]),
        },
    }
    with open(out_dir/f"res_{suffix}.json","w") as f:
        json.dump(results, f, indent=4)
    model.save_weights(out_dir/f"weights_{suffix}.weights.h5")
    print(f"DONE: {suffix}")

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

