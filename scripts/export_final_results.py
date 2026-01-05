import os
import sys
import json
import warnings
import importlib
import numpy as np
import joblib
import tensorflow as tf
from pyvista import ImageData 

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# path config
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(script_dir)

def safe_model_import(folder_name, module_names):
    path = os.path.join(script_dir, folder_name)
    sys.path.insert(0, path)
    modules = []
    for name in module_names:
        if name in sys.modules:
            del sys.modules[name] 
        modules.append(importlib.import_module(name))
    sys.path.remove(path)
    return modules

try:
    pi_core, pi_val = safe_model_import("PI_DeepONet", ["core", "validation"])
    dn_core, dn_val = safe_model_import("DeepONet", ["core", "validation"])
    pinn_core, pinn_val = safe_model_import("PINN", ["core_pinn", "validation_pinn"])
    
    sys.path.insert(0, os.path.join(script_dir, "STA_LSTM"))
    from core_sta_lstm import load_data_sta, STALSTM
    print("modules loaded")
except Exception as e:
    print(f"error in loading modules: {e}")
    sys.exit(1)

GRIDS = [50, 100, 250, 500]
CYTOKINES = ["il8", "il1", "il6", "il10", "tnf", "tgf"]
SEED = 42 

WINDOWS = {
    "Window_82_100": range(82, 101),
    "Window_72_89": range(72, 90)
}

def get_best_params(model_folder, res_str):
    path = os.path.join(ROOT_PATH, "models", model_folder, res_str, "research_report.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("best_params", {"hidden_size": 128, "activation": "tanh", "latent_dim": 64, "lr": 0.0001})
    return {"hidden_size": 128, "activation": "tanh", "latent_dim": 64, "lr": 0.0001}

def save_vtk(data_grid, step, output_dir, grid_size):
    grid = ImageData()
    grid.dimensions = (grid_size, grid_size, 1)
    grid.spacing = (1, 1, 1)
    for i, name in enumerate(CYTOKINES):
        grid.point_data[name] = data_grid[:, :, i].flatten(order="F")
    os.makedirs(output_dir, exist_ok=True)
    grid.save(os.path.join(output_dir, f"step_{step:03d}.vtk"))

def export_all():
    os.chdir(ROOT_PATH)
    
    for res in GRIDS:
        res_str = f"{res}x{res}"
        print(f"\nsize: {res_str}")
        
        # BASELINE
        y_true_path = os.path.join("preprocessed", res_str, "Y_target.npy")
        if os.path.exists(y_true_path):
            print("Baseline...")
            y_true = np.load(y_true_path)
            for w_name, w_range in WINDOWS.items():
                out_dir = os.path.join(f"cc3d_export_s42/BASELINE", res_str, w_name)
                for t in w_range:
                    if 0 <= t-2 < len(y_true): save_vtk(y_true[t-2], t, out_dir, res)

        # GPR
        gpr_path = os.path.join("models/gpr", res_str, f"model_seed_{SEED}.joblib")
        if os.path.exists(gpr_path):
            print("GPR...")
            model_gpr = joblib.load(gpr_path)
            coords_raw = np.load(os.path.join("preprocessed", res_str, "X_trunk.npy")).astype(np.float32)
            coords = coords_raw[0] if coords_raw.ndim == 3 else coords_raw
            for w_name, w_range in WINDOWS.items():
                out_dir = os.path.join(f"cc3d_export_s42/GPR", res_str, w_name)
                for t in w_range:
                    X_in = np.hstack([coords, np.full((coords.shape[0], 1), t)])
                    save_vtk(model_gpr.predict(X_in).reshape(res, res, 6), t, out_dir, res)

        # DEEPONET
        dn_weights = os.path.join("models/deeponet_dde", res_str, f"model_seed_{SEED}-5000.weights.h5")
        if os.path.exists(dn_weights):
            print("DEEPONET...")
            tf.keras.backend.clear_session()
            train, val, test, coords = dn_core.load_data_deeponet(res)
            params = get_best_params("deeponet_dde", res_str)
            
            train_dn = (train[0], coords, train[1])
            val_dn = (val[0], coords, val[1])
            model = dn_val.create_model(params, train_dn, val_dn, train[0].shape[1], coords.shape[1])
            
            model.compile("adam", lr=params.get('lr', 0.0001))
            
            model.net((train[0][:1], coords))
            model.net.load_weights(dn_weights, skip_mismatch=True)
            for w_name, w_range in WINDOWS.items():
                out_dir = os.path.join(f"cc3d_export_s42/DEEPONET", res_str, w_name)
                for t in w_range:
                    t_idx = t - 82
                    if 0 <= t_idx < len(test[0]):
                        pred = model.predict((test[0][t_idx:t_idx+1], coords))
                        save_vtk(np.asarray(pred).reshape(res, res, 6), t, out_dir, res)

        # PI-DEEPONET
        pi_weights = os.path.join("models/pi_deeponet_dde", res_str, f"model_seed_{SEED}-5000.weights.h5")
        if os.path.exists(pi_weights):
            print("PI-DEEPONET...")
            tf.keras.backend.clear_session()
            train, val, test, coords = pi_core.load_data_pideeponet(res)
            params = get_best_params("pi_deeponet_dde", res_str)
            model = pi_val.create_pideeponet_model(params, res, train, val, coords)
            
            model.compile("adam", lr=params.get('lr', 0.0001))
            
            model.net((train[0][:1], coords))
            model.net.load_weights(pi_weights, skip_mismatch=True)
            for w_name, w_range in WINDOWS.items():
                out_dir = os.path.join(f"cc3d_export_s42/PI_DEEPONET", res_str, w_name)
                for t in w_range:
                    t_idx = t - 82
                    if 0 <= t_idx < len(test[0]):
                        pred = model.predict((test[0][t_idx:t_idx+1], coords))
                        save_vtk(np.asarray(pred).reshape(res, res, 6), t, out_dir, res)

        # PINN
        pinn_weights = os.path.join("models/pinn", res_str, f"model_seed_{SEED}-10000.weights.h5")
        if os.path.exists(pinn_weights):
            print("PINN...")
            tf.keras.backend.clear_session()
            train, val, test = pinn_core.load_data_pinn(res)
            params = get_best_params("pinn", res_str)
            model_pinn, _, _ = pinn_val.create_pinn_model(params, res, train, val)
            
            model_pinn.compile("adam", lr=params.get('lr', 0.0001))
            
            model_pinn.net(train[0][:1]); model_pinn.net.load_weights(pinn_weights, skip_mismatch=True)
            x_ax = np.linspace(0, 1, res); y_ax = np.linspace(0, 1, res)
            gx, gy = np.meshgrid(x_ax, y_ax); coords_2d = np.column_stack((gx.ravel(), gy.ravel()))
            for w_name, w_range in WINDOWS.items():
                out_dir = os.path.join(f"cc3d_export_s42/PINN", res_str, w_name)
                for t in w_range:
                    X_pts = np.hstack([coords_2d, np.full((coords_2d.shape[0], 1), t)]).astype(np.float32)
                    pred = model_pinn.predict(X_pts)
                    save_vtk(np.asarray(pred).reshape(res, res, 6), t, out_dir, res)

        # STA-LSTM
        lstm_path = os.path.join("models/sta_lstm", res_str, f"model_seed_{SEED}.h5")
        if os.path.exists(lstm_path):
            print("STA-LSTM...")
            tf.keras.backend.clear_session()
            _, _, test = load_data_sta(res)
            params = get_best_params("sta_lstm", res_str)
            model_lstm = STALSTM(params['hidden_size'], (res, res, 6), res, params.get('activation', 'ReLU'))
            model_lstm(test[0][:1]) 
            
            model_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('lr', 0.0001)), loss='mse')
            
            model_lstm.load_weights(lstm_path)
            for w_name, w_range in WINDOWS.items():
                out_dir = os.path.join(f"cc3d_export_s42/STA_LSTM", res_str, w_name)
                for t in w_range:
                    t_idx = t - 80
                    if 0 <= t_idx < len(test[0]):
                        pred = model_lstm.predict(test[0][t_idx:t_idx+1], verbose=0)[0]
                        save_vtk(pred, t, out_dir, res)

if __name__ == "__main__":
    export_all()
