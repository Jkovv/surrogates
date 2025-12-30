import deepxde as dde
import numpy as np
import tensorflow as tf
from core import get_scaled_physics

def create_model(params, grid_size, train_data):
    X_b, coords, y = train_data
    D_init, k_init = get_scaled_physics(grid_size)
    
    D_var = [dde.Variable(np.float32(d)) for d in D_init]
    k_var = [dde.Variable(np.float32(k)) for k in k_init]

    def pde(x, y):
        res = []
        for i in range(6):
            u_t = dde.grad.jacobian(y, x, i=i, j=2)
            u_xx = dde.grad.hessian(y, x, component=i, i=0, j=0)
            u_yy = dde.grad.hessian(y, x, component=i, i=1, j=1)
            res.append(u_t - D_var[i] * (u_xx + u_yy) + k_var[i] * y[:, i:i+1])
        return res

    net = dde.nn.DeepONetCartesianProd(
        [X_b.shape[1], params['hidden_size'], params['hidden_size'], params['latent_dim']],
        [2, params['hidden_size'], params['hidden_size'], params['latent_dim']],
        params['activation'], "Glorot uniform",
        num_outputs=6, multi_output_strategy="independent"
    )
    net.apply_output_transform(lambda x, y: tf.nn.relu(y))

    data = dde.data.TripleCartesianProd(
        X_train=(X_b, coords), y_train=y,
        X_test=(X_b, coords), y_test=y
    )
    
    return dde.Model(data, net), D_var, k_var