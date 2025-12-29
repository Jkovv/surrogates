import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class SpatialTemporalAttention(layers.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_s = layers.Dense(hidden_size)
        self.W_t = layers.Dense(hidden_size)
        self.V = layers.Dense(1)

    def call(self, lstm_output, flattened_input):
        spatial_attention = tf.tanh(self.W_s(lstm_output))
        temporal_attention = tf.tanh(self.W_t(flattened_input))
        attention_scores = self.V(spatial_attention * temporal_attention)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        attended_output = tf.matmul(tf.transpose(attention_weights, [0, 2, 1]), lstm_output)
        return attended_output

class STALSTM(models.Model):
    def __init__(self, hidden_size, target_shape, grid_size, act_name="ReLU"):
        super().__init__()
        acts = {"ReLU": "relu", "Tanh": "tanh", "SiLU": "swish", "GELU": "gelu"}
        activation = acts.get(act_name, "relu")

        pool_factor = grid_size // 50
        if pool_factor > 1:
            self.pool = layers.AveragePooling3D(pool_size=(1, pool_factor, pool_factor))
        else:
            self.pool = layers.Lambda(lambda x: x)

        self.projection = layers.Dense(128, activation=activation)
        self.lstm = layers.LSTM(hidden_size, return_sequences=True)
        self.attention = SpatialTemporalAttention(hidden_size)
        self.fc1 = layers.Dense(128, activation=activation) 
        self.batch_norm = layers.BatchNormalization()
        self.fc_out = layers.Dense(np.prod(target_shape), activation='linear', dtype='float32')
        self.reshape_layer = layers.Reshape(target_shape, dtype='float32')

    def call(self, input_data):
        x = self.pool(input_data)
        s = x.shape
        x_flat = tf.reshape(x, (-1, s[1], s[2] * s[3] * s[4]))
        
        x_proj = self.projection(x_flat)
        x_lstm = self.lstm(x_proj)
        x_att = self.attention(x_lstm, x_proj)
        
        x_out_flat = tf.reshape(x_att, (-1, x_att.shape[-1]))
        x = self.fc1(x_out_flat)
        x = self.batch_norm(x)
        x = self.fc_out(x)
        return self.reshape_layer(x)

def load_data_sta(grid_size):
    path = f"preprocessed/{grid_size}x{grid_size}"
    X = np.load(os.path.join(path, "X_lstm.npy")).astype(np.float32) 
    Y = np.load(os.path.join(path, "Y_target.npy")).astype(np.float32)
    n = X.shape[0]
    return (X[:int(n*0.7)], Y[:int(n*0.7)]), (X[int(n*0.7):int(n*0.8)], Y[int(n*0.7):int(n*0.8)]), (X[int(n*0.8):], Y[int(n*0.8):])