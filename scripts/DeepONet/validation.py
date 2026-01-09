import deepxde as dde
import tensorflow as tf
import gc
import numpy as np

def mse_3d(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())

def create_model(params, train_data, val_data, b_dim, t_dim, seed):
    data = dde.data.TripleCartesianProd(
        X_train=(train_data[0], train_data[1]), y_train=train_data[2],
        X_test=(val_data[0], val_data[1]), y_test=val_data[2]
    )

    np.random.seed(seed)
    num_f = 128 
    B_np = np.random.normal(scale=15.0, size=(t_dim, num_f)).astype(np.float32)
    B_fixed = tf.constant(B_np)

    def fourier_transform(x):
        projection = 2.0 * np.pi * tf.matmul(x, B_fixed)
        return tf.concat([tf.sin(projection), tf.cos(projection)], axis=-1)

    layer_sizes_branch = [b_dim, params['hidden_size'], params['hidden_size'], params['latent_dim']]
    layer_sizes_trunk = [num_f * 2, params['hidden_size'], params['hidden_size'], params['latent_dim']]

    net = dde.nn.DeepONetCartesianProd(
        layer_sizes_branch, layer_sizes_trunk, params['activation'], "Glorot uniform",
        num_outputs=6, multi_output_strategy="independent"
    )

    for i in range(len(net.trunk)):
        net.trunk[i].apply_feature_transform(fourier_transform)

    net.apply_output_transform(lambda x, y: tf.nn.relu(y))
    model = dde.Model(data, net)
    return model

def train_and_eval(params, train, val, b_dim, t_dim, seed):
    tf.keras.backend.clear_session(); gc.collect()
    dde.config.set_random_seed(seed)
    model = create_model(params, train, val, b_dim, t_dim, seed)
    model.compile("adam", lr=params['lr'], metrics=[mse_3d, r2_score])
    losshistory, train_state = model.train(iterations=params['epochs'], display_every=1000)
    return train_state, model
