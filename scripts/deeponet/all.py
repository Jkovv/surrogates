import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
import optuna
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed):
    s = int(seed)
    os.environ['PYTHONHASHSEED'] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

def positional_encoding(coords, L=6):
    out = [coords]
    for i in range(L):
        for fn in [tf.sin, tf.cos]:
            out.append(fn(2.0**i * np.pi * coords))
    return tf.concat(out, axis=-1)

def build_conv_deeponet(grid_size, n_features, n_filters, p_dim, L):
    init = tf.keras.initializers.HeNormal()

    # BRANCH NET
    branch_in = tf.keras.layers.Input(shape=(grid_size, grid_size, n_features))
    b = tf.keras.layers.Conv2D(n_filters, 3, padding='same')(branch_in)
    b = tf.keras.layers.LeakyReLU(0.2)(b)
    b = tf.keras.layers.MaxPooling2D(2)(b)
    b = tf.keras.layers.Conv2D(n_filters * 2, 3, padding='same')(b)
    b = tf.keras.layers.LeakyReLU(0.2)(b)
    b = tf.keras.layers.Flatten()(b)
    branch_out = tf.keras.layers.Dense(p_dim, activation='tanh')(b)

    # TRUNK NET
    trunk_raw = tf.keras.layers.Input(shape=(3,))
    t_encoded = tf.keras.layers.Lambda(lambda x: positional_encoding(x, L=L))(trunk_raw)
    t = tf.keras.layers.Dense(256)(t_encoded)
    t = tf.keras.layers.LeakyReLU(0.2)(t)
    t = tf.keras.layers.LayerNormalization()(t)
    t = tf.keras.layers.Dense(256)(t)
    t = tf.keras.layers.LeakyReLU(0.2)(t)
    trunk_out = tf.keras.layers.Dense(p_dim, activation='tanh')(t)

    combined = tf.keras.layers.Multiply()([branch_out, trunk_out])
    
    output = tf.keras.layers.Dense(1, activation='softplus',
                                   bias_initializer=tf.keras.initializers.Constant(0.01))(combined)
    
    return tf.keras.Model(inputs=[branch_in, trunk_raw], outputs=output)

def calculate_metrics(y_true, y_pred, masks):
    yt, yp = y_true.flatten(), y_pred.flatten()
    r2 = r2_score(yt, yp)
    
    # Masked RMSE
    spatial_mask = np.max(masks, axis=-1).squeeze()
    denom = np.sum(spatial_mask) + 1e-7
    rmse = np.sqrt(np.sum(np.square(y_true - y_pred) * spatial_mask) / denom)
    
    # Spatial Correlation
    if np.std(yp) > 1e-9:
        pearson, _ = pearsonr(yt, yp)
    else:
        pearson = 0.0
    
    # DICE
    thresh = 0.01 
    y_t_b, y_p_b = (y_true > thresh), (y_pred > thresh)
    dice = (2. * np.sum(y_t_b * y_p_b)) / (np.sum(y_t_b) + np.sum(y_p_b) + 1e-7)

    # SSIM
    ssim_list = []
    for i in range(len(y_true)):
        dr = max(y_true[i].max(), 1.0)
        ssim_list.append(ssim(y_true[i], y_pred[i], data_range=dr))
    
    return {
        "Global_R2": float(r2),
        "Masked_RMSE": float(rmse),
        "Avg_Dice": float(dice),
        "Avg_SSIM": float(np.mean(ssim_list)),
        "Spatial_Correlation": float(pearson)
    }

def run_experiment(grid, cytokine, seed):
    set_seed(seed)
    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir = Path("./models/deeponet")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_b = np.load(data_path / "X_branch.npy")
    X_t = np.load(data_path / "X_trunk.npy")
    Y_all = np.load(data_path / "Y_target.npy")
    M_all = np.load(data_path / "Y_masks.npy")
    
    idx = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}[cytokine]
    n_samples = X_b.shape[0]
    train_idx, val_idx = int(0.7 * n_samples), int(0.8 * n_samples)
    bs = 256 

    def make_dataset(start, end):
        def gen():
            for i in range(start, end):
                b_tile = np.tile(X_b[i:i+1], (grid*grid, 1, 1, 1))
                yield (b_tile, X_t[i]), Y_all[i, ..., idx].flatten()[:, np.newaxis]
        return tf.data.Dataset.from_generator(
            gen, output_signature=((tf.TensorSpec(shape=(grid*grid, grid, grid, X_b.shape[-1]), dtype=tf.float32),
                                    tf.TensorSpec(shape=(grid*grid, 3), dtype=tf.float32)),
                                   tf.TensorSpec(shape=(grid*grid, 1), dtype=tf.float32))
        ).unbatch().batch(bs).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = make_dataset(0, train_idx), make_dataset(train_idx, val_idx)

    def objective(trial):
        tf.keras.backend.clear_session()
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 4e-3, log=True),
            "p_dim": trial.suggest_categorical("p_dim", [128, 256]),
            "n_filters": trial.suggest_categorical("n_filters", [32, 64]),
            "L": trial.suggest_int("L", 4, 8)
        }
        model = build_conv_deeponet(grid, X_b.shape[-1], params['n_filters'], params['p_dim'], params['L'])
        model.compile(optimizer=tf.keras.optimizers.Adam(params['lr']), loss='mse')
        model.fit(train_ds.take(100), epochs=5, verbose=0)
        return model.evaluate(val_ds.take(50), verbose=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    best = study.best_params

    model = build_conv_deeponet(grid, X_b.shape[-1], best['n_filters'], best['p_dim'], best['L'])
    model.compile(optimizer=tf.keras.optimizers.Adam(best['lr']), loss='mse')
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
    
    preds = []
    for i in range(n_samples):
        p = model.predict([np.tile(X_b[i:i+1], (grid*grid, 1, 1, 1)), X_t[i]], verbose=0)
        preds.append(p.reshape(grid, grid))
    preds = np.array(preds)

    res = {
        "params": {
            "filters": best['n_filters'],
            "p_dim": best['p_dim'],
            "L": best['L'],
            "lr": best['lr']
        },
        "seed": seed,
        "grid": grid,
        "cytokine": cytokine,
        "results": {
            "Interpolation_72_89": calculate_metrics(Y_all[70:87, ..., idx], preds[70:87], M_all[70:87]),
            "Extrapolation_82_100": calculate_metrics(Y_all[80:98, ..., idx], preds[80:98], M_all[80:98])
        }
    }
    
    suffix = f"results_deeponet_{cytokine}_{grid}_s{seed}"
    model.save_weights(out_dir / f"weights_{suffix}.weights.h5")
    with open(out_dir / f"{suffix}.json", 'w') as f: json.dump(res, f, indent=4)
    print(f">>> DeepONet experiment finished. Results unified with STA-LSTM structure.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(args.grid, args.cytokine.lower(), args.seed)
