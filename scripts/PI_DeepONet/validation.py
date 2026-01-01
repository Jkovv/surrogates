import deepxde as dde
import tensorflow as tf
import gc
from core import get_scaled_physics

def create_pideeponet_model(params, grid_size, train_raw, val_raw, coords):
    D_phys, k_phys = get_scaled_physics(grid_size)
    data = dde.data.TripleCartesianProd(
        X_train=(train_raw[0], coords), y_train=train_raw[1],
        X_test=(val_raw[0], coords), y_test=val_raw[1]
    )
    net = dde.nn.DeepONetCartesianProd(
        [train_raw[0].shape[1], params['hidden_size'], params['hidden_size'], params['latent_dim']],
        [2, params['hidden_size'], params['hidden_size'], params['latent_dim']],
        params['activation'], "Glorot uniform",
        num_outputs=6, multi_output_strategy="independent"
    )
    net.apply_output_transform(lambda x, y: tf.nn.relu(y))

    def pde(x, y):
        res = []
        for i in range(6):
            u_xx = dde.grad.hessian(y, x, component=i, i=0, j=0)
            u_yy = dde.grad.hessian(y, x, component=i, i=1, j=1)
            res.append(- D_phys[i] * (u_xx + u_yy) + k_phys[i] * y[:, i:i+1])
        return res

    model = dde.Model(data, net)
    return model