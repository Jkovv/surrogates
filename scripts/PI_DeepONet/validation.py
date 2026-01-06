import deepxde as dde
import tensorflow as tf
import gc
from core import get_scaled_physics

def create_pideeponet_model(params, train_data, val_data):
    physics = get_scaled_physics()
    
    data = dde.data.TripleCartesianProd(
        X_train=(train_data[0], train_data[1]), y_train=train_data[2],
        X_test=(val_data[0], val_data[1]), y_test=val_data[2]
    )
    
    net = dde.nn.DeepONetCartesianProd(
        [train_data[0].shape[1], params['hidden_size'], params['hidden_size'], params['latent_dim']],
        [train_data[1].shape[1], params['hidden_size'], params['hidden_size'], params['latent_dim']],
        params['activation'], "Glorot uniform",
        num_outputs=6, multi_output_strategy="independent"
    )
    
    def pde_res(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        return - physics["diff_coeffs"] * (dy_xx + dy_yy) + physics["decay_coeffs"] * y

    net.apply_output_transform(lambda x, y: tf.nn.relu(y))
    model = dde.Model(data, net)
    return model

def train_and_eval(params, train_data, val_data, seed):
    tf.keras.backend.clear_session(); gc.collect()
    dde.config.set_random_seed(seed)
    
    model = create_pideeponet_model(params, train_data, val_data)
    model.compile("adam", lr=params['lr'])
    
    losshistory, train_state = model.train(iterations=params['epochs'], display_every=1000)
    return train_state, model
