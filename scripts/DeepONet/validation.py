import deepxde as dde

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

    # non-negative concentrations
    net.apply_output_transform(lambda x, y: dde.backend.tf.nn.relu(y))

    model = dde.Model(data, net)
    return model

def train_and_eval(params, train_data, val_data, b_dim, t_dim, seed):
    dde.config.set_random_seed(seed)
    model = create_model(params, train_data, val_data, b_dim, t_dim)
    model.compile("adam", lr=params['lr'])
    
    _, train_state = model.train(iterations=params['epochs'])
    return train_state.best_loss[1], model