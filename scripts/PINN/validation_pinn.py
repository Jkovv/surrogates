import deepxde as dde
import numpy as np
from core_pinn import get_scaled_physics

def create_pinn_model(params, grid_size, train_data, val_data, initial_physics=None):
    X_train, y_train = train_data
    X_val, y_val = val_data

    # physics "discovery"
    if initial_physics is None:
        D_init, k_init = get_scaled_physics(grid_size)
    else:
        D_init, k_init = np.array(initial_physics['D']), np.array(initial_physics['k'])
    
    # inverse problem (todo: fix?)
    D_var = [dde.Variable(d) for d in D_init]
    k_var = [dde.Variable(k) for k in k_init]

    def pde(x, y):
        residuals = []
        for i in range(6):
            # i=cytokine index, j=input coordinate index (t is at index 2)
            u_t = dde.grad.jacobian(y, x, i=i, j=2)
            u_xx = dde.grad.hessian(y, x, component=i, i=0, j=0)
            u_yy = dde.grad.hessian(y, x, component=i, i=1, j=1)
            
            res = u_t - D_var[i] * (u_xx + u_yy) + k_var[i] * y[:, i:i+1]
            residuals.append(res)
        return residuals

    data = dde.data.DataSet(X_train, y_train, X_val, y_val, check_priority=True)

    net = dde.nn.FNN(
        [3] + [params['hidden_size']] * 3 + [6],
        params['activation'],
        "Glorot uniform"
    )

    # enforcing non-negativity of concentrations
    net.apply_output_transform(lambda x, y: dde.backend.tf.nn.relu(y))

    model = dde.Model(data, net)
    return model, D_var, k_var