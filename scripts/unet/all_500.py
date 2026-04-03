import argparse, json, gc, os, time
import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass

GRID         = 500
RESULTS_DIR  = "models/unet"
CYT_MAP      = {"il8": 0, "il10": 3}
FULL_EPOCHS  = 200
TUNE_EPOCHS  = 20
N_OPTUNA     = 20
PATIENCE     = 20

def build_unet(base_filters, depth, dropout_rate):
    inputs = tf.keras.Input(shape=(GRID, GRID, 22))
    skips  = []
    x = inputs
    f = base_filters
    for d in range(depth):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        f *= 2
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    for d in reversed(range(depth)):
        f //= 2
        x = tf.keras.layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        s = skips[d]
        x = tf.keras.layers.Lambda(lambda t: tf.concat([t[0], t[1][:, :tf.shape(t[0])[1], :tf.shape(t[0])[2], :]], axis=-1))([x, s])
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Conv2D(1, 1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def do_train(model, opt, Xu_mmap, Yt, cyt_idx, tr_idx, vl_idx, batch_size, epochs, verbose=True):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y - pred))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x, y):
        return tf.reduce_mean(tf.square(y - model(x, training=False)))

    def get_batch(idx):
        raw = Xu_mmap[idx].astype(np.float32)
        # Reshape X_unet (B, G, G, 22)
        if raw.ndim == 5: # if mmap (B, 2, G, G, 11)
            x = raw.transpose(0, 2, 3, 1, 4).reshape(len(idx), GRID, GRID, 22)
        else: # if (B, G, G, 22)
            x = raw
        y = Yt[idx, :, :, cyt_idx:cyt_idx+1].astype(np.float32)
        return x, y

    x_v, y_v = get_batch(vl_idx)
    best_val, stagnant, best_w = 1e9, 0, None
    for epoch in range(1, epochs + 1):
        np.random.shuffle(tr_idx)
        epoch_losses = []
        for s in range(0, len(tr_idx), batch_size):
            x_b, y_b = get_batch(tr_idx[s:s+batch_size])
            epoch_losses.append(train_step(x_b, y_b))
        vl = float(val_step(x_v, y_v).numpy())
        if verbose and epoch % 5 == 0:
            print(f"  Ep {epoch:3d} | Loss: {np.mean(epoch_losses):.6f} | Val: {vl:.6f}", flush=True)
        if vl < best_val:
            best_val, stagnant, best_w = vl, 0, model.get_weights()
        else:
            stagnant += 1
            if stagnant >= PATIENCE: break
    if best_w: model.set_weights(best_w)
    return best_val

def run(cyt_name, seed, data_dir):
    print(f"\nUNet 500x500 (A100) | cytokine={cyt_name} | seed={seed}")
    np.random.seed(seed); tf.random.set_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cyt_idx = CYT_MAP[cyt_name]
    md = json.load(open(f"{data_dir}/metadata.json"))
    clip = float(md["scaling"]["max"][cyt_idx])

    Xu_mmap = np.load(f"{data_dir}/X_unet.npy", mmap_mode="r")
    Yt      = np.load(f"{data_dir}/Y_target.npy")
    Yraw    = np.load(f"{data_dir}/Y_raw_phys.npy")

    def parse_split(s):
        start, end = s.split(':')
        return np.arange(int(start), int(end))

    tr_idx = parse_split(md["splits"]["train"])
    vl_idx = parse_split(md["splits"]["val"])
    ts_near_idx = parse_split(md["splits"]["test_near"])
    ts_far_idx  = parse_split(md["splits"]["test_far"])
    
    print(f"Splits: Train={len(tr_idx)}, Val={len(vl_idx)}, Near={len(ts_near_idx)}, Far={len(ts_far_idx)}")

    if seed == 42:
        def objective(trial):
            tf.keras.backend.clear_session()
            f = trial.suggest_categorical("base_filters", [16, 32])
            d = trial.suggest_categorical("depth", [2, 3])
            dr = trial.suggest_categorical("dropout", [0.0, 0.1])
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            bs = trial.suggest_categorical("batch_size", [2, 4])
            m = build_unet(f, d, dr); opt = tf.keras.optimizers.Adam(lr)
            return do_train(m, opt, Xu_mmap, Yt, cyt_idx, tr_idx, vl_idx, bs, TUNE_EPOCHS, verbose=False)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_OPTUNA); best = study.best_params
    else:
        best = json.load(open(f"{RESULTS_DIR}/res_{cyt_name}_500_42.json"))["best_params"]

    tf.keras.backend.clear_session()
    model = build_unet(best["base_filters"], best["depth"], best["dropout"])
    opt = tf.keras.optimizers.Adam(best["lr"])
    t_start = time.time()
    do_train(model, opt, Xu_mmap, Yt, cyt_idx, tr_idx, vl_idx, best["batch_size"], FULL_EPOCHS)
    train_elapsed = time.time() - t_start

    def evaluate(indices):
        preds = []
        for idx in indices:
            raw = Xu_mmap[idx:idx+1].astype(np.float32)
            if raw.ndim == 5:
                xi = raw.transpose(0, 2, 3, 1, 4).reshape(1, GRID, GRID, 22)
            else: xi = raw
            preds.append(model(xi, training=False).numpy()[0, :, :, 0])
        p_phys = np.clip((np.array(preds) + 1.0) / 2.0 * clip, 0, None)
        gt_phys = Yraw[indices + 1, ..., cyt_idx]
        
        r2 = float(r2_score(gt_phys.flatten(), p_phys.flatten()))
        rmse = float(np.sqrt(np.mean((gt_phys - p_phys)**2)))
        ssim_v = float(np.mean([ssim(gt_phys[t], p_phys[t], data_range=clip) for t in range(len(indices)) if gt_phys[t].std() > 1e-12]))
        return {"Global_R2": r2, "Unmasked_RMSE": rmse, "SSIM": ssim_v}

    res = {
        "grid": 500, "seed": seed, "cytokine": cyt_name, "best_params": best,
        "train_time_seconds": round(train_elapsed, 2),
        "results": {
            "Near_Horizon": evaluate(ts_near_idx),
            "Far_Horizon": evaluate(ts_far_idx)
        }
    }
    out_path = f"{RESULTS_DIR}/res_{cyt_name}_500_{seed}.json"
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cytokine", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", default="preprocessed_200h/500x500")
    args = ap.parse_args()
    run(args.cytokine, args.seed, args.data_dir)
