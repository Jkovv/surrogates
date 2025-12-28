import deepxde as dde
import numpy as np

def create_model(params, train_data, val_data, b_dim, t_dim):
    data = dde.data.Triple(
        X_train=train_data[0], y_train=train_data[1],
        X_test=val_data[0], y_test=val_data[1]
    )

    net = dde.nn.DeepONet(
        [b_dim, params['hidden_size'], params['hidden_size'], params['latent_dim'] * 6],
        [t_dim, params['hidden_size'], params['hidden_size'], params['latent_dim']],
        params['activation'], "Glorot uniform"
    )

    net.apply_output_transform(lambda x, y: dde.backend.tf.nn.relu(y))

    model = dde.Model(data, net)
    
    def pde_loss(x, y):
        # x[1] is the trunk input (coordinates)
        res = []
        for i in range(6):
            u_xx = dde.grad.hessian(y, x, component=i, i=0, j=0)
            u_yy = dde.grad.hessian(y, x, component=i, i=1, j=1)
            res.append(u_xx + u_yy)
        return res

    model.add_physics(pde_loss) 
    
    return model

def train_and_eval(params, train_data, val_data, b_dim, t_dim, seed):
    dde.config.set_random_seed(seed)
    model, pde_fn = create_model(params, train_data, val_data, b_dim, t_dim)
    
    model.compile("adam", lr=params['lr'], loss_weights=[1.0, params['pde_weight']])
    
    _, train_state = model.train(iterations=params['epochs'])
    return train_state.best_loss[1], model