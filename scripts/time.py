import os, json, argparse, time, warnings
from pathlib import Path
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CYTOKINE_NAMES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
EVAL_CHUNK = 4096
N_WARMUP = 2
N_RUNS = 5

# model definitions for loading weights

# STA-LSTM
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.attn_conv = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")
    def call(self, x): return x * self.attn_conv(x)

class STALSTM(tf.keras.Model):
    def __init__(self, grid_size, filters=64, lstm_units=128):
        super().__init__()
        ls = max(grid_size // 4, 8)
        self.enc = tf.keras.layers.TimeDistributed(tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu"),
        ]), name="encoder")
        self.spatial_attn = tf.keras.layers.TimeDistributed(SpatialAttention(), name="spatial_attention")
        self.gap = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D(), name="gap")
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=False, name="lstm")
        self.relu = tf.keras.layers.Activation("relu")
        self.fc = tf.keras.layers.Dense(ls * ls * filters, activation="relu")
        self.reshape_latent = tf.keras.layers.Reshape((ls, ls, filters))
        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters//2, 3, strides=2, padding="same", activation="relu")
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters//4, 3, strides=2, padding="same", activation="relu")
        self.out_conv = tf.keras.layers.Conv2D(1, 3, padding="same", activation="linear")
        self.out_resize = tf.keras.layers.Resizing(grid_size, grid_size)
    def call(self, x):
        h = self.enc(x); h = self.spatial_attn(h); h = self.gap(h)
        h = self.lstm(h); h = self.relu(h); h = self.fc(h); h = self.reshape_latent(h)
        return self.out_resize(self.out_conv(self.deconv2(self.deconv1(h))))

# DeepONet
class Branch(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.fc2 = tf.keras.layers.Dense(p, activation="linear")
    def call(self, x, training=False): return self.fc2(self.fc1(x))

class Trunk(tf.keras.layers.Layer):
    def __init__(self, hidden, p, **kw):
        super().__init__(**kw)
        self.U = tf.keras.layers.Dense(hidden, activation="tanh")
        self.V = tf.keras.layers.Dense(hidden, activation="tanh")
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

class DeepONetModel(tf.keras.Model):
    def __init__(self, hidden, p):
        super().__init__()
        self.branch = Branch(hidden, p)
        self.trunk = Trunk(hidden, p)
        self.bias = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="bias")
    def call(self, inputs, training=False):
        xb, xt = inputs
        b = self.branch(xb, training=training)
        t = self.trunk(xt)
        return tf.expand_dims(tf.einsum("bp,bnp->bn", b, t) + self.bias, -1)

# U-Net
def _conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def _encoder_block(x, filters):
    skip = _conv_block(x, filters)
    return skip, tf.keras.layers.MaxPooling2D(2, padding="same")(skip)

def _decoder_block(x, skip, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same", activation="relu")(x)
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        x = tf.keras.layers.Resizing(skip.shape[1], skip.shape[2])(x)
    return _conv_block(tf.keras.layers.Concatenate()([x, skip]), filters)

def build_unet(grid_size, in_channels=22, base_filters=32, depth=4, dropout=0.0):
    inputs = tf.keras.Input(shape=(grid_size, grid_size, in_channels))
    skips = []; x = inputs
    for i in range(depth):
        skip, x = _encoder_block(x, base_filters * (2**i)); skips.append(skip)
    x = _conv_block(x, base_filters * (2**depth))
    if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
    for i in reversed(range(depth)):
        x = _decoder_block(x, skips[i], base_filters * (2**i))
    return tf.keras.Model(inputs, tf.keras.layers.Conv2D(1, 1, padding="same", activation="linear")(x))

# DeepONet feature builders
def build_branch_inputs(Xb, Xt, cyt_idx):
    N, _, G, _, _ = Xb.shape
    f0 = Xb[:, 0, :, :, cyt_idx]
    mask = (Xb[:, 0, :, :, 6:].max(axis=-1) > 0.5).astype(np.float32)
    xs = np.linspace(0, 1, G, dtype=np.float32)
    ys = np.linspace(0, 1, G, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    out = np.zeros((N, 7), dtype=np.float32)
    for i in range(N):
        f = f0[i]; m = mask[i]; na = float(np.sum(m)) + 1e-6
        out[i, 0] = (float(np.max(f)) + 1.0) / 2.0
        out[i, 1] = (float(np.mean(f)) + 1.0) / 2.0
        out[i, 2] = float(np.std(f))
        out[i, 3] = float(np.sum(xx * m) / na)
        out[i, 4] = float(np.sum(yy * m) / na)
        out[i, 5] = na / (G * G)
        out[i, 6] = float(Xt[i, 0, 2])
    return out

def build_trunk_inputs(Xb, Xt):
    N, _, G, _, C = Xb.shape
    vals = Xb.transpose(0, 2, 3, 1, 4).reshape(N, G*G, 22).astype(np.float32)
    return np.concatenate([Xt[:, :, :2].astype(np.float32), vals], axis=-1)


# pred functions

def predict_sta_lstm(model, X):
    return model.predict(X, batch_size=2, verbose=0)

def predict_unet(model, X):
    return model.predict(X, batch_size=2, verbose=0)

def predict_deeponet(model, Xbranch, Xtrunk):
    N, n_pts, _ = Xtrunk.shape
    out = np.zeros((N, n_pts, 1), np.float32)
    for i in range(N):
        xb = tf.constant(Xbranch[i:i+1])
        for s in range(0, n_pts, EVAL_CHUNK):
            e = min(s + EVAL_CHUNK, n_pts)
            out[i, s:e] = model([xb, tf.constant(Xtrunk[i:i+1, s:e])], training=False).numpy()[0]
    return out

def predict_pinn(model, xy_grid, G2, N):
    t_norm_all = np.linspace(-1, 1, 101, dtype=np.float64)
    out = np.zeros((N, G2), np.float64)
    for i in range(N):
        tt = np.full((G2, 1), float(t_norm_all[i + 2]), np.float64)
        out[i] = model.predict(np.hstack([xy_grid, tt])).flatten()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, choices=["sta_lstm", "deeponet", "pinn", "unet"])
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--cytokine", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    idx = CYTOKINE_NAMES.index(args.cytokine.lower())
    grid = args.grid
    G2 = grid * grid
    data_path = Path(f"./preprocessed/{grid}x{grid}")
    suffix = f"{args.cytokine}_{grid}_{args.seed}"

    model_dirs = {
        "sta_lstm": "models/sta_lstm",
        "deeponet": "models/deeponet_h",
        "pinn": "models/pinn",
        "unet": "models/unet",
    }
    out_dir = Path(model_dirs[args.model])
    res_path = out_dir / f"res_{suffix}.json"

    with open(res_path) as f:
        bp = json.load(f)["best_params"]

    print(f"Benchmarking {args.model.upper()} | {grid}x{grid} | {args.cytokine} | seed {args.seed}")
    print(f"  HP: {bp}")

    # load model + weights
    if args.model == "sta_lstm":
        X = np.load(data_path / "X_lstm.npy").astype(np.float32)
        model = STALSTM(grid_size=grid, filters=bp["filters"], lstm_units=bp["lstm_units"])
        model(X[:1])
        model.load_weights(str(out_dir / f"weights_{suffix}.weights.h5"))
        def do_predict(): return predict_sta_lstm(model, X)

    elif args.model == "unet":
        X = np.load(data_path / "X_unet.npy").astype(np.float32)
        model = build_unet(grid, in_channels=X.shape[-1],
                           base_filters=bp["base_filters"], depth=bp["depth"],
                           dropout=bp.get("dropout", 0.0))
        model(X[:1])
        model.load_weights(str(out_dir / f"weights_{suffix}.weights.h5"))
        def do_predict(): return predict_unet(model, X)

    elif args.model == "deeponet":
        Xb = np.load(data_path / "X_branch.npy").astype(np.float32)
        Xt = np.load(data_path / "X_trunk.npy").astype(np.float32)
        Xbranch = build_branch_inputs(Xb, Xt, idx)
        Xtrunk = build_trunk_inputs(Xb, Xt)
        model = DeepONetModel(hidden=bp["hidden"], p=bp["p"])
        dummy_xb = tf.constant(Xbranch[:1])
        dummy_xt = tf.constant(Xtrunk[:1, :2])
        _ = model([dummy_xb, dummy_xt], training=False)
        model.load_weights(str(out_dir / f"weights_{suffix}.weights.h5"))
        def do_predict(): return predict_deeponet(model, Xbranch, Xtrunk)

    elif args.model == "pinn":
        import deepxde as dde
        dde.config.set_default_float("float64")
        dde.config.disable_xla_jit()

        Y_tgt = np.load(data_path / "Y_target.npy").astype(np.float64)
        M_pinn = np.load(data_path / "Y_masks_pinn.npy").astype(np.float64)
        Y_ic = np.load(data_path / "Y_ic.npy").astype(np.float64)
        with open(data_path / "metadata.json") as f:
            meta = json.load(f)
        clip_max = float(np.array(meta["scaling"]["max"])[idx])
        N = Y_tgt.shape[0]

        TRUE_SIZE = 5.0; S_MCS = 60.0; H_MCS = 1.0 / S_MCS
        MASK_E, MASK_NDN, MASK_NA, MASK_M1, MASK_M2 = 0, 1, 2, 3, 4
        areaconv = TRUE_SIZE**2 / grid**2; volumeconv = areaconv
        D_all = np.array([2.09e-6,3e-7,8.49e-8,1.45e-8,4.07e-9,2.6e-7]) * S_MCS / areaconv
        k_all = np.array([.2,.6,.5,.5,.5*.225,.5/25]) * H_MCS
        sec = np.array([234e-5,1.46e-5,3.024e-5,225e-5,250e-5,45e-5,250e-5,70e-5,280e-5]) * volumeconv * H_MCS
        D = float(D_all[idx]); k = float(k_all[idx])
        masks_mean = M_pinn[:70].mean(axis=0)
        me=masks_mean[:,0]; mnn=masks_mean[:,1]; mna=masks_mean[:,2]
        mm1=masks_mean[:,3]; mm2=masks_mean[:,4]; z=np.zeros(G2,np.float64)
        s1_map=[sec[0]*me,sec[3]*mna,sec[4]*mm1,sec[5]*mm2,sec[6]*mna,sec[8]*mm2]
        s2_map=[sec[1]*mnn,z,z,z,sec[7]*mm1,z]
        e_map=[sec[2]*mna,z,z,z,z,z]
        sc=2.0/(clip_max+1e-30)
        s1_tf=tf.constant(s1_map[idx].reshape(-1,1)*sc,tf.float64)
        s2_tf=tf.constant(s2_map[idx].reshape(-1,1)*sc,tf.float64)
        e_tf=tf.constant(e_map[idx].reshape(-1,1),tf.float64)

        G_tf=tf.constant(grid,dtype=tf.float64)
        D_tf=tf.constant([[D]],dtype=tf.float64)
        k_tf=tf.constant([[k]],dtype=tf.float64)
        def pde_fn(x, y):
            d2x=dde.grad.hessian(y,x,i=0,j=0); d2y_=dde.grad.hessian(y,x,i=1,j=1)
            ut=dde.grad.jacobian(y,x,i=0,j=2)
            G_dyn=tf.cast(G_tf,tf.float64)
            ix=tf.cast(tf.clip_by_value(tf.floor((x[:,0:1]+1)/2*G_dyn),0,G_dyn-1),tf.int32)
            iy=tf.cast(tf.clip_by_value(tf.floor((x[:,1:2]+1)/2*G_dyn),0,G_dyn-1),tf.int32)
            fi=tf.squeeze(ix*tf.cast(G_tf,tf.int32)+iy,axis=1)
            return ut-(D_tf*(d2x+d2y_)-k_tf*y+tf.gather(s1_tf,fi)+tf.gather(s2_tf,fi)-tf.gather(e_tf,fi)*y)

        t_norm_all=np.linspace(-1,1,101,dtype=np.float64)
        t_min=float(t_norm_all[2]); t_max=float(t_norm_all[-1])
        geom=dde.geometry.Rectangle([-1,-1],[1,1])
        geomtime=dde.geometry.GeometryXTime(geom,dde.geometry.TimeDomain(t_min,t_max))

        xs=np.linspace(-1,1,grid,dtype=np.float64); ys=np.linspace(-1,1,grid,dtype=np.float64)
        xx,yy=np.meshgrid(xs,ys,indexing="ij")
        xy_grid=np.stack([xx.ravel(),yy.ravel()],axis=1)

        X_ic=np.hstack([xy_grid,np.full((G2,1),t_min,np.float64)])
        Y_ic_obs=Y_ic[:,:,idx].reshape(G2,1).astype(np.float64)

        ic_bc=dde.PointSetBC(X_ic,Y_ic_obs,component=0)
        neu_bc=dde.NeumannBC(geomtime,lambda x:np.zeros((len(x),1),np.float64),
                             lambda x,on_boundary:on_boundary,component=0)
        obs_bc=dde.PointSetBC(X_ic,Y_ic_obs,component=0)

        data = dde.data.TimePDE(geomtime,pde_fn,ic_bcs=[ic_bc,neu_bc,obs_bc],
                                num_domain=100,num_boundary=100,num_initial=0,
                                train_distribution="uniform")
        net=dde.maps.FNN([3]+[bp["hidden"]]*bp["n_layers"]+[1],"tanh","Glorot uniform")
        net.apply_output_transform(lambda x,y:y)
        pinn_model=dde.Model(data,net)
        pinn_model.compile("adam",lr=1e-3)

        _ = pinn_model.predict(np.hstack([xy_grid[:2],np.zeros((2,1),np.float64)]))

        import glob
        ckpt_prefix = str(out_dir / f"ckpt_{suffix}")
        wgt_prefix = str(out_dir / f"weights_{suffix}")
        candidates = []
        for f in sorted(glob.glob(str(out_dir / "*.weights.h5"))):
            name = Path(f).name
            if name.startswith(f"ckpt_{suffix}") or name.startswith(f"weights_{suffix}"):
                try:
                    step = int(name.split("-")[-1].replace(".weights.h5", ""))
                    candidates.append((step, f))
                except ValueError:
                    candidates.append((0, f))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            print(f"Loading: {Path(candidates[0][1]).name}")
            pinn_model.net.load_weights(candidates[0][1])
        else:
            print(f"WARNING: no weights found, using random init")

        model = pinn_model
        def do_predict(): return predict_pinn(model, xy_grid, G2, N)

    # Warmup
    print(f"Warmup ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        do_predict()

    # Timed runs
    print(f"Timing ({N_RUNS} runs)...")
    times = []
    for r in range(N_RUNS):
        t0 = time.time()
        do_predict()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"Run {r+1}: {elapsed:.3f}s")

    mean_pred = float(np.mean(times))
    std_pred = float(np.std(times))

    # loading train_time_seconds from res json (this was train+pred combined)
    with open(res_path) as f:
        res_data = json.load(f)
    total_time = res_data.get("train_time_seconds", None)
    if total_time is not None:
        train_only = max(total_time - mean_pred, 0.0)
    else:
        train_only = None

    result = {
        "model": args.model,
        "grid": grid,
        "cytokine": args.cytokine,
        "seed": args.seed,
        "n_samples": 99,
        "train_time_seconds": round(train_only, 2) if train_only is not None else None,
        "pred_time_seconds": round(mean_pred, 4),
        "pred_time_std": round(std_pred, 4),
        "pred_per_sample_ms": round(mean_pred / 99 * 1000, 4),
        "total_time_seconds": round(total_time, 2) if total_time is not None else None,
        "pred_runs": [round(t, 4) for t in times],
    }

    out_path = out_dir / f"time_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n  Pred:  {result['pred_time_seconds']:.3f}s ± {result['pred_time_std']:.3f}s ({result['pred_per_sample_ms']:.2f}ms/sample)")
    if train_only is not None:
        print(f"  Train: {result['train_time_seconds']:.1f}s")
        print(f"  Total: {result['total_time_seconds']:.1f}s (from res json)")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()