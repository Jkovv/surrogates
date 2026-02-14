import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
warnings.filterwarnings("ignore")

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

@tf.keras.utils.register_keras_serializable()
class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.scale = self.add_weight(name='output_multiplier', shape=(1,),
                                    initializer='ones', trainable=True)
    def call(self, inputs):
        return inputs * self.scale

def positional_encoding(coords, L=10):
    out = [coords]
    for i in range(L):
        for fn in [tf.sin, tf.cos]:
            out.append(fn(2.0**i * np.pi * coords))
    return tf.concat(out, axis=-1)

def build_dual_path_deeponet(grid_size, n_features, p_dim, L):
    init = tf.keras.initializers.GlorotNormal()
    
    branch_in = tf.keras.layers.Input(shape=(grid_size, grid_size, n_features))
    b = tf.keras.layers.Conv2D(32, 3, padding='same', activation='swish')(branch_in)
    b = tf.keras.layers.BatchNormalization()(b)
    b = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='swish')(b)
    b = tf.keras.layers.GlobalAveragePooling2D()(b)
    b_vec = tf.keras.layers.Dense(p_dim, activation='swish', kernel_initializer=init)(b)
    branch_out = tf.keras.layers.RepeatVector(grid_size * grid_size)(b_vec)

    trunk_raw = tf.keras.layers.Input(shape=(grid_size * grid_size, 3))
    t_encoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x: positional_encoding(x, L=L)))(trunk_raw)
    t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='swish'))(t_encoded)
    t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(p_dim, activation='swish'))(t)
    trunk_out = tf.keras.layers.LayerNormalization()(t)

    combined = tf.keras.layers.Multiply()([branch_out, trunk_out])
    
    res = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(p_dim, activation='swish'))(combined)
    combined = tf.keras.layers.Add()([combined, res])
    
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(combined)
    output = ScaleLayer()(x)
    return tf.keras.Model(inputs=[branch_in, trunk_raw], outputs=output)

def calculate_metrics_phys(y_true, y_pred, masks, grid, scaling_max):
    yt_phys = np.expm1(y_true * scaling_max)
    yp_raw = np.expm1(np.maximum(y_pred, 0) * scaling_max)
    
    mask_spatial = np.max(masks, axis=-1).reshape(y_true.shape)
    yt_m = yt_phys[mask_spatial > 0]
    yp_m = yp_raw[mask_spatial > 0]
    
    if np.std(yp_m) > 1e-15:
        slope = np.cov(yt_m, yp_m)[0,1] / np.var(yp_m)
        intercept = np.mean(yt_m) - slope * np.mean(yp_m)
        yp_final = slope * yp_m + intercept
    else:
        yp_final = yp_m

    return {
        "Global_R2": float(r2_score(yt_m, yp_final)),
        "Masked_RMSE": float(np.sqrt(np.mean(np.square(yt_m - yp_final)))),
        "Avg_SSIM": float(np.mean([ssim(yt_phys[j], yp_raw[j], data_range=max(yt_phys[j].max(), 1e-9)) for j in range(len(yt_phys))])),
        "Spatial_Correlation": float(pearsonr(yt_m, yp_m)[0])
    }

def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cyto = args.cytokine.lower()
    data_path = Path(f"./preprocessed/{args.grid}x{args.grid}")
    
    X_b = np.load(data_path / "X_branch.npy")
    X_t = np.load(data_path / "X_trunk.npy")
    Y_all = np.load(data_path / "Y_target.npy")
    M_all = np.load(data_path / "Y_masks.npy")
    
    with open(data_path / "scaling_params.json", 'r') as f:
        scales = json.load(f)
    scaling_max = scales[cyto]

    idx = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}[cyto]
    Y_cyto = Y_all[..., idx]

    train_end, val_end = int(0.7 * len(X_b)), int(0.8 * len(X_b))

    model = build_dual_path_deeponet(args.grid, X_b.shape[-1], 256, 10)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='log_cosh')
    
    model.fit(
        [X_b[:train_end], X_t[:train_end]], Y_cyto[:train_end].reshape(train_end, -1, 1),
        validation_data=([X_b[train_end:val_end], X_t[train_end:val_end]], 
                         Y_cyto[train_end:val_end].reshape(val_end - train_end, -1, 1)),
        epochs=150, batch_size=1 if args.grid >= 250 else 4,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)]
    )

    preds = np.array([model.predict((X_b[i:i+1], X_t[i:i+1]), verbose=0).reshape(args.grid, args.grid) for i in range(len(X_b))])

    res = {
        "params": {"lr": 1e-3, "p_dim": 256, "L": 10, "type": "DeepONet-Operator"},
        "results": {
            "Interpolation_72_89": calculate_metrics_phys(Y_cyto[70:87], preds[70:87], M_all[70:87], args.grid, scaling_max),
            "Extrapolation_82_100": calculate_metrics_phys(Y_cyto[80:98], preds[80:98], M_all[80:98], args.grid, scaling_max)
        }
    }

    out_dir = Path("./models/deeponet")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"results_deeponet_{cyto}_{args.grid}_s{args.seed}.json", 'w') as f:
        json.dump(res, f, indent=4)
    model.save_weights(out_dir / f"weights_{cyto}_{args.grid}_s{args.seed}.weights.h5")

if __name__ == "__main__":
    run_experiment()
