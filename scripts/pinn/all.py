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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
warnings.filterwarnings("ignore")

CYTOKINE_MAP = {"il8": 0, "il1": 1, "il6": 2, "il10": 3, "tnf": 4, "tgf": 5}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class PINN(tf.keras.Model):
    def __init__(self, layers=[128, 128, 128, 128]):
        super().__init__()
        # FNN
        self.fcs = [tf.keras.layers.Dense(l, activation='swish', kernel_initializer='glorot_normal') for l in layers]
        self.out_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = inputs
        for layer in self.fcs:
            x = layer(x)
        return self.out_layer(x)

def calculate_metrics_phys(y_true, y_pred, masks, grid_size):
    spatial_mask = np.max(masks, axis=-1, keepdims=True) 
    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    r2 = r2_score(y_true_f, y_pred_f)
    
    # Masked RMSE
    sq_diff = np.square(y_true - y_pred) * spatial_mask
    m_rmse = np.sqrt(np.sum(sq_diff) / (np.sum(spatial_mask) + 1e-7))

    corrs, ssim_vals = [], []
    for t in range(y_true.shape[0]):
        gt, pr = y_true[t, :, :, 0], y_pred[t, :, :, 0]
        # SSIM
        ssim_vals.append(ssim(gt, pr, data_range=max(gt.max(), 1.0)))
        # corr
        if np.std(gt) > 1e-9 and np.std(pr) > 1e-9:
            corrs.append(pearsonr(gt.flatten(), pr.flatten())[0])

    return {
        "Global_R2": float(r2),
        "Masked_RMSE": float(m_rmse),
        "Avg_SSIM": float(np.mean(ssim_vals)),
        "Spatial_Correlation": float(np.mean(corrs)) if corrs else 0.0
    }

def run_pinn_experiment(grid, cytokine, seed):
    set_seed(seed)
    cyto = cytokine.lower()
    data_path = Path(f"./preprocessed/{grid}x{grid}")
    out_dir = Path("./models/pinn")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"pinn_{cyto}_grid{grid}_seed{seed}"
    
    X_coords = np.load(data_path / "X_trunk.npy").astype(np.float32) # [Samples, Grid*Grid, 3]
    Y_all_scaled = np.load(data_path / "Y_target.npy").astype(np.float32) # [Samples, Grid, Grid, 6]
    M_all = np.load(data_path / "Y_masks.npy").astype(np.float32)

    with open(data_path / "scaling_params.json", "r") as f:
        scaling_params = json.load(f)
    max_val_log = tf.constant(scaling_params[cyto], dtype=tf.float32)

    idx = CYTOKINE_MAP[cyto]
    Y_target_cyto = Y_all_scaled[..., idx:idx+1] # [Samples, Grid, Grid, 1]

    # 70/10/20
    n_samples = len(X_coords)
    train_end = int(0.7 * n_samples)
    val_end = int(0.8 * n_samples)

    X_train = X_coords[:train_end].reshape(-1, 3)
    Y_train = Y_target_cyto[:train_end].reshape(-1, 1)

    model = PINN()
    optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_batch)
            y_pred = model(x_batch) 
            
            loss_data = tf.reduce_mean(tf.square(y_batch - y_pred))
            
            u_phys = tf.exp(y_pred * max_val_log) - 1.0
            
            grads = tape.gradient(u_phys, x_batch)
            u_t = grads[:, 2:3]
            
            loss_phys = tf.reduce_mean(tf.square(u_t)) # d_u/dt = 0
            
            total_loss = loss_data + 1e-4 * loss_phys
            
        grads_model = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads_model, model.trainable_variables))
        return total_loss, loss_data

    batch_size = 8192 if grid >= 250 else 2048
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(10000).batch(batch_size)

    print(f"starting training: {suffix}")
    for epoch in range(50): 
        for x_b, y_b in dataset:
            tl, ld = train_step(x_b, y_b)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss: {tl:.6f}")

    Y_pred_log = []
    for i in range(n_samples):
        p = model.predict(X_coords[i], batch_size=batch_size, verbose=0)
        Y_pred_log.append(p.reshape(grid, grid, 1))
    
    Y_pred_log = np.array(Y_pred_log)
    
    # expm1(pred * max_log)
    Y_pred_phys = np.expm1(Y_pred_log * float(max_val_log))
    Y_true_phys = np.expm1(Y_target_cyto * float(max_val_log))

    res = {
        "params": {"model": "PINN", "layers": "4x128", "grid": grid, "seed": seed, "cytokine": cyto},
        "results": {
            "Interpolation_72_89": calculate_metrics_phys(Y_true_phys[70:88], Y_pred_phys[70:88], M_all[70:88], grid),
            "Extrapolation_82_100": calculate_metrics_phys(Y_true_phys[80:99], Y_pred_phys[80:99], M_all[80:99], grid)
        }
    }

    json_path = out_dir / f"results_{suffix}.json"
    weights_path = out_dir / f"weights_{suffix}.weights.h5"
    
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=4)
    model.save_weights(weights_path)
    
    print(f"Saved JSON: {json_path}")
    print(f"Saved Weights: {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--cytokine", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_pinn_experiment(args.grid, args.cytokine, args.seed)
