import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import os
import random
import deepxde as dde
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Set default floating-point type
tf.keras.backend.set_floatx('float32')

# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Configure TensorFlow for deterministic behavior
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Control threading for reproducibility
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Ensure GPU determinism
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define He initializer
initializer = tf.keras.initializers.HeNormal(seed=seed_value)

sorted_concatenated_csv = "C:/Users/johnd/Documents/GitHub/neural-agent-models/data/simulation_data/50x50.csv"
data = pd.read_csv(sorted_concatenated_csv)
#data.drop(columns=['zCOM'], inplace=True)
print(data.head())
data['time'] = (data['mcsteps'] / 10000).astype(int)
data = data[['time'] + [col for col in data.columns if col != 'time']]
data.drop(columns=['mcsteps'], inplace=True)
print(data)
cytokine_columns = ['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']
smallest_values = data[cytokine_columns].min()
largest_values = data[cytokine_columns].max()

print("Smallest values for each cytokine:")
print(smallest_values)
print("\nLargest values for each cytokine:")
print(largest_values)

# define cytokines
cytokines = ['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']

# Remove brackets and convert to float
for col in cytokines:
    data[col] = data[col].str.strip('[]').astype(float)

# get unique time values
unique_time = data['time'].unique()

arrays = {}

# iterate over unique time values
for time in unique_time:
    # filter data for current value of time
    data_time = data[data['time'] == time]
    
    array = np.zeros((50, 50, len(cytokines)))
    
    # get X and Y coordinates
    x = data_time['xCOM'].astype(int)
    y = data_time['yCOM'].astype(int)
    
    # get cytokine concentrations
    concentrations = data_time[['il8', 'il1', 'il6', 'il10', 'tnf', 'tgf']].values
    
    # assign cytokine concentrations to corresponding position in array
    array[x, y, :] = concentrations
    
    # store array for current value of time
    arrays[time] = array

sequence_length = 10
input_sequences = []
output_values = []

# convert dictionary values to a list of arrays
arrays_list = [arrays[key] for key in sorted(arrays.keys())]

# convert 'arrays' list to numpy array
arrays_np = np.array(arrays_list)

for i in range(len(arrays_np) - sequence_length):
    input_seq = arrays_np[i:i+sequence_length]  # input sequence of arrays
    output_val = arrays_np[i+sequence_length]   # array at next time step
    
    input_sequences.append(input_seq)
    output_values.append(output_val)

# convert lists to numpy arrays
input_sequences = np.array(input_sequences)
output_values = np.array(output_values)

print(input_sequences.shape)
print(output_values.shape)

# Simulation parameters
nx = 50
true_size = 5
s_mcs = 60.0
h_mcs = 1 / 60.0
lineconv = true_size / nx
areaconv = true_size**2 / nx**2
volumeconv = (true_size**2 * 1) / (nx**2 * 1)

# Parameters for each cytokine
Dil8 = 2.09e-6 * s_mcs / areaconv
muil8 = 0.2 * h_mcs
keil8 = 234e-5 * volumeconv * h_mcs
kndnil8 = 1.46e-5 * volumeconv * h_mcs
thetanail8 = 3.024e-5 * volumeconv * h_mcs

Dil1 = 3e-7 * s_mcs / areaconv
muil1 = 0.6 * h_mcs
knail1 = 225e-5 * volumeconv * h_mcs

Dil6 = 8.49e-8 * s_mcs / areaconv
muil6 = 0.5 * h_mcs
km1il6 = 250e-5 * volumeconv * h_mcs

Dil10 = 1.45e-8 * s_mcs / areaconv
muil10 = 0.5 * h_mcs
km2il10 = 45e-5 * volumeconv * h_mcs

Dtnf = 4.07e-9 * s_mcs / areaconv
mutnf = 0.5 * 0.225 * h_mcs
knatnf = 250e-5 * volumeconv * h_mcs
km1tnf = 70e-5 * volumeconv * h_mcs

Dtgf = 2.6e-7 * s_mcs / areaconv
mutgf = 0.5 * (1 / 25) * h_mcs
km2tgf = 280e-5 * volumeconv * h_mcs

cellpresente = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
cellpresentndn = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
cellpresentna = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
cellpresentm1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
cellpresentm2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print("cellpresente:", cellpresente.shape)
print("cellpresentndn:", cellpresentndn.shape)
print("cellpresentna:", cellpresentna.shape)
print("cellpresentm1:", cellpresentm1.shape)
print("cellpresentm2:", cellpresentm2.shape)

# Convert the NumPy arrays to TensorFlow constants
cellpresente_tf = tf.constant(cellpresente, dtype=tf.float32)
cellpresentndn_tf = tf.constant(cellpresentndn, dtype=tf.float32)
cellpresentna_tf = tf.constant(cellpresentna, dtype=tf.float32)
cellpresentm1_tf = tf.constant(cellpresentm1, dtype=tf.float32)
cellpresentm2_tf = tf.constant(cellpresentm2, dtype=tf.float32)

# Define the parameters as TensorFlow constants
keil8_tf = tf.constant(keil8, dtype=tf.float32)
knail1_tf = tf.constant(knail1, dtype=tf.float32)
km1il6_tf = tf.constant(km1il6, dtype=tf.float32)
km2il10_tf = tf.constant(km2il10, dtype=tf.float32)
knatnf_tf = tf.constant(knatnf, dtype=tf.float32)
km2tgf_tf = tf.constant(km2tgf, dtype=tf.float32)
kndnil8_tf = tf.constant(kndnil8, dtype=tf.float32)
km1tnf_tf = tf.constant(km1tnf, dtype=tf.float32)
thetanail8_tf = tf.constant(thetanail8, dtype=tf.float32)

# diffusion degradation secretion endocytosis
D = np.array([Dil8, Dil1, Dil6, Dil10, Dtnf, Dtgf])
k = np.array([muil8, muil1, muil6, muil10, mutnf, mutgf])

# Stack the parameters for secretion and endocytosis
s1 = tf.stack([
    keil8_tf * cellpresente_tf,
    knail1_tf * cellpresentna_tf,
    km1il6_tf * cellpresentm1_tf,
    km2il10_tf * cellpresentm1_tf,
    knatnf_tf * cellpresentna_tf,
    km2tgf_tf * cellpresentm2_tf
], axis=0)

s2 = tf.stack([
    kndnil8_tf * cellpresentndn_tf,
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    km1tnf_tf * cellpresentm1_tf,
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32)
], axis=0)

e = tf.stack([
    thetanail8_tf * cellpresentna_tf,
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32),
    tf.zeros_like(cellpresentndn_tf, dtype=tf.float32)
], axis=0)

print("s1:", s1)
print("s2:", s2)
print("e:", e)


print("D:", D)
print("k:", k)
print("s1:", s1)
print("s2:", s2)
print("e:", e)

T_MAX = int(arrays_np.shape[0] - 1)

n_times = T_MAX + 1  # 91
train_end = int(0.7 * n_times) - 1
val_end   = int(0.8 * n_times) - 1

train_times = np.arange(0, train_end + 1)
val_times   = np.arange(train_end + 1, val_end + 1)
test_times  = np.arange(val_end + 1, T_MAX + 1)

print("train_times:", train_times[0], "...", train_times[-1], "len=", len(train_times))
print("val_times:", val_times[0], "...", val_times[-1], "len=", len(val_times))
print("test_times:", test_times[0], "...", test_times[-1], "len=", len(test_times))

# TF constants for PDE
D_tf = tf.constant(D.reshape(1, 6), dtype=tf.float32)
k_tf = tf.constant(k.reshape(1, 6), dtype=tf.float32)

def pde(x, y):
    """
    x: (N,3) -> [x, y, t]
    y: (N,6) -> cytokines (il8, il1, il6, il10, tnf, tgf)

    Enforces:
        u_t = D * (u_xx + u_yy) - k*u + s1(t) + s2(t) - e(t)*u

    So residual is:
        r = u_t - ( D*laplacian - k*u + s1 + s2 - e*u ) = 0
    """

    u = [y[:, i:i+1] for i in range(6)]

    #spatial Laplacian for each cytokine
    lap = []
    for i in range(6):
        d2x = dde.grad.hessian(u[i], x, i=0, j=0)
        d2y = dde.grad.hessian(u[i], x, i=1, j=1)
        lap.append(d2x + d2y)

    laplacian_u = tf.concat(lap, axis=1)  # (N,6)

    # time derivative u_t for each cytokine
    ut = []
    for i in range(6):
        ut_i = dde.grad.jacobian(u[i], x, i=0, j=2)
        ut.append(ut_i)
    u_t = tf.concat(ut, axis=1)

    degradation = k_tf * y

    t = x[:, 2]
    time_idx = tf.cast(tf.round(t), tf.int32)

    max_idx = tf.shape(tf.transpose(s1))[0] - 1
    max_idx = tf.cast(max_idx, tf.int32)

    time_idx = tf.where(time_idx < 0, tf.zeros_like(time_idx), time_idx)
    time_idx = tf.where(time_idx > max_idx, tf.fill(tf.shape(time_idx), max_idx), time_idx)

    s1_t = tf.gather(tf.transpose(s1), time_idx)
    s2_t = tf.gather(tf.transpose(s2), time_idx)
    e_t  = tf.gather(tf.transpose(e),  time_idx)
    endocytosis = e_t * y
    rhs = laplacian_u * D_tf - degradation + s1_t + s2_t - endocytosis
    return u_t - rhs

# Geometry/time domain
geom = dde.geometry.Rectangle([0, 0], [49, 49])
timedomain = dde.geometry.TimeDomain(0, T_MAX)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

class SaveTrainingDataCallback(dde.callbacks.Callback):
    def __init__(self, filename, interval=1000):
        self.filename = filename
        self.interval = interval
        self.history = []
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self):
        print("Starting training...")

    def get_learning_rate(self):
        try:
            return self.model.opt.learning_rate.numpy()
        except AttributeError:
            try:
                return self.model.opt.lr.numpy()
            except AttributeError:
                return None

    def on_epoch_end(self):
        if self.model.train_state.epoch % self.interval == 0:
            current_data = {
                'epoch': self.model.train_state.epoch,
                'train_loss': self.model.train_state.loss_train,
                'test_loss': self.model.train_state.loss_test,
                'metrics': self.model.train_state.metrics_test,
                'lr': self.get_learning_rate()
            }
            self.history.append(current_data)
            np.save(self.filename, self.history)
            print(f"Saved training data at epoch {self.model.train_state.epoch}")

    def on_train_end(self):
        np.save(self.filename, self.history)
        print("Training finished and data saved.")

nx = 50
coords = np.stack(np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)

X_obs_list = []
Y_obs_list = []

n_points_per_t = 200
rng = np.random.default_rng(seed_value)

for t in train_times:
    idx = rng.choice(coords.shape[0], size=min(n_points_per_t, coords.shape[0]), replace=False)
    xy = coords[idx]  # (K,2)
    tt = np.full((xy.shape[0], 1), t, dtype=np.float32)
    X_obs_list.append(np.hstack([xy, tt]))

    u = arrays_np[t].reshape(-1, 6).astype(np.float32)
    Y_obs_list.append(u[idx])

X_obs = np.vstack(X_obs_list)
Y_obs = np.vstack(Y_obs_list)

# Zero-flux Neumann BC for each component
def on_boundary(x, on_boundary):
    return on_boundary

zero = lambda x: np.zeros((len(x), 1), dtype=np.float32)

bcs = [dde.NeumannBC(geomtime, zero, on_boundary, component=i) for i in range(6)]

# Observation constraints
obs_bcs = [
    dde.PointSetBC(X_obs, Y_obs[:, i:i+1], component=i)
    for i in range(6)
]

# Data-driven initial condition at t=0
t0 = 0
use_all_ic_points = True
n_ic_points = 2500

if use_all_ic_points:
    ic_idx = np.arange(coords.shape[0])
else:
    ic_idx = rng.choice(coords.shape[0], size=min(n_ic_points, coords.shape[0]), replace=False)

xy0 = coords[ic_idx]  # (K,2)
tt0 = np.full((xy0.shape[0], 1), t0, dtype=np.float32)
X_ic = np.hstack([xy0, tt0]).astype(np.float32)  # (K,3)

Y0 = arrays_np[t0].reshape(-1, 6).astype(np.float32)[ic_idx]  # (K,6)

ic_data_bcs = [
    dde.PointSetBC(X_ic, Y0[:, i:i+1], component=i)
    for i in range(6)
]

data = dde.data.TimePDE(
    geomtime,
    pde,
    ic_bcs=ic_data_bcs + bcs + obs_bcs,
    num_domain=500,
    num_boundary=200,
    num_initial=100,
)

net = dde.maps.FNN([3] + [50] * 3 + [6], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: tf.nn.softplus(y))

model = dde.Model(data, net)
save_training_data_callback = SaveTrainingDataCallback(
    filename="PINN(50x50)(72)_train_data.npy",
    interval=100,
)

model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=8000, display_every=1000, callbacks=[save_training_data_callback])

X_rand = geomtime.random_points(5000)
res = model.predict(X_rand, operator=pde)
print("Residual mean abs:", np.mean(np.abs(res)))
print("Residual max abs:", np.max(np.abs(res)))
print("Residual per-cyt mean:", np.mean(np.abs(res), axis=0))

print("Number of loss terms:", len(train_state.loss_train))
for i, v in enumerate(train_state.loss_train):
    print(i, v)

def make_grid_xy():
    xs, ys = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

xy_grid = make_grid_xy()

def eval_times(times):
    mses, maes = [], []
    for t in times:
        tt = np.full((xy_grid.shape[0], 1), t, dtype=np.float32)
        X = np.hstack([xy_grid, tt])
        Y_true = arrays_np[t].reshape(-1, 6).astype(np.float32)
        Y_pred = model.predict(X)
        mses.append(np.mean((Y_true - Y_pred) ** 2))
        maes.append(np.mean(np.abs(Y_true - Y_pred)))
    return float(np.mean(mses)), float(np.mean(maes))

val_mse, val_mae = eval_times(val_times)
print(f"HELD-OUT VAL (times {val_times[0]}..{val_times[-1]}): MSE={val_mse:.3e}, MAE={val_mae:.3e}")

test_mse, test_mae = eval_times(test_times)
print(f"HELD-OUT TEST (times {test_times[0]}..{test_times[-1]}): MSE={test_mse:.3e}, MAE={test_mae:.3e}")

def rel_mae_nonzero(times, eps=1e-8, thresh=1e-6):
    rels = []
    for t in times:
        tt = np.full((xy_grid.shape[0], 1), t, dtype=np.float32)
        X = np.hstack([xy_grid, tt])
        Y_true = arrays_np[t].reshape(-1, 6).astype(np.float32)
        Y_pred = model.predict(X)

        mask = np.abs(Y_true) > thresh
        if np.any(mask):
            rels.append(np.mean(np.abs(Y_true[mask] - Y_pred[mask]) / (np.abs(Y_true[mask]) + eps)))
    return float(np.mean(rels)) if rels else np.nan

def nmae(times, eps=1e-8):
    vals=[]
    for t in times:
        tt=np.full((xy_grid.shape[0],1), t, np.float32)
        X=np.hstack([xy_grid, tt])
        Y_true=arrays_np[t].reshape(-1,6).astype(np.float32)
        Y_pred=model.predict(X)
        denom=np.mean(np.abs(Y_true)) + eps
        vals.append(np.mean(np.abs(Y_true - Y_pred))/denom)
    return float(np.mean(vals))

print("VAL relMAE(thresh=1e-4):", rel_mae_nonzero(val_times, thresh=1e-4))
print("TEST relMAE(thresh=1e-4):", rel_mae_nonzero(test_times, thresh=1e-4))

losses = np.array(train_state.loss_train, dtype=np.float64)
print("PDE loss:", losses[0])
print("BC+IC+OBS total:", losses[1:].sum())
print("PDE fraction:", losses[0] / losses.sum())