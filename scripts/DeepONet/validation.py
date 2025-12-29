import deepxde as dde
import tensorflow as tf
import gc

def mse_3d(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def create_model(params, train_data, val_data, b_dim, t_dim):
    data = dde.data.TripleCartesianProd(
        X_train=(train_data[0], train_data[1]), y_train=train_data[2],
        X_test=(val_data[0], val_data[1]), y_test=val_data[2]
    )
    net = dde.nn.DeepONetCartesianProd(
        [b_dim, params['hidden_size'], params['hidden_size'], params['latent_dim']],
        [t_dim, params['hidden_size'], params['hidden_size'], params['latent_dim']],
        params['activation'], "Glorot uniform",
        num_outputs=6, multi_output_strategy="independent"
    )
    net.apply_output_transform(lambda x, y: tf.nn.relu(y))
    model = dde.Model(data, net)
    return model

def train_and_eval(params, train, val, b_dim, t_dim, seed):
    tf.keras.backend.clear_session()
    gc.collect()
    dde.config.set_random_seed(seed)
    model = create_model(params, train, val, b_dim, t_dim)
    model.compile("adam", lr=params['lr'], metrics=[mse_3d])
    losshistory, train_state = model.train(iterations=params['epochs'], display_every=1000)
    return train_state.best_metrics[0], model