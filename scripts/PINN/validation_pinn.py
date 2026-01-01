import deepxde as dde
import numpy as np
import tensorflow as tf
from core_pinn import get_scaled_physics

def create_pinn_model(params, grid_size, train_data, val_data):
    X_train_full, y_train_full = train_data
    num_anchors = min(len(X_train_full), 20000)
    idx = np.random.choice(len(X_train_full), num_anchors, replace=False)
    X_train, y_train = X_train_full[idx], y_train_full[idx]

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

    bcs = [dde.icbc.PointSetBC(X_train, y_train[:, i:i+1], component=i) for i in range(6)]
    geomtime = dde.geometry.GeometryXTime(dde.geometry.Rectangle([0, 0], [1, 1]), dde.geometry.TimeDomain(0, 100))
    data = dde.data.TimePDE(geomtime, pde, bcs, num_domain=2000, anchors=X_train, num_test=1000)
    
    net = dde.nn.FNN([3] + [params['hidden_size']] * 3 + [6], params['activation'], "Glorot uniform")
    net.apply_output_transform(lambda x, y: tf.nn.relu(y))

    return dde.Model(data, net), D_var, k_var
