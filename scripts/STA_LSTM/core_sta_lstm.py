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

    def call(self, lstm_output, input_data):
        # attention scores
        spatial_attention = tf.tanh(self.W_s(lstm_output))
        temporal_attention = tf.tanh(self.W_t(input_data))
        attention_scores = self.V(spatial_attention * temporal_attention)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # weights -> LSTM output
        attended_output = tf.matmul(tf.transpose(attention_weights, [0, 2, 1]), lstm_output)
        return attended_output

class STALSTM(models.Model):
    def __init__(self, hidden_size, target_shape, act_name="ReLU"):
        super().__init__()
        acts = {"ReLU": "relu", "Tanh": "tanh", "SiLU": "swish", "GELU": "gelu"}
        activation = acts.get(act_name, "relu")

        self.hidden_size = hidden_size
        self.lstm = layers.LSTM(hidden_size, return_sequences=True)
        self.attention = SpatialTemporalAttention(hidden_size)
        self.fc1 = layers.Dense(128, activation=activation) 
        self.batch_norm = layers.BatchNormalization()
        self.fc_out = layers.Dense(np.prod(target_shape), activation='linear')
        self.reshape = layers.Reshape(target_shape)

    def call(self, input_data):
        x = self.lstm(input_data)
        x = self.attention(x, input_data)
        x = tf.reshape(x, (-1, self.hidden_size))
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.fc_out(x)
        return self.reshape(x)

def load_data_sta(grid_size):
    path = f"preprocessed/{grid_size}x{grid_size}"
    # loading pre-processed sequence data
    X = np.load(os.path.join(path, "X_lstm.npy")) 
    Y = np.load(os.path.join(path, "Y_target.npy"))
    
    n_samples = X.shape[0]
    val_split_idx = int(n_samples * 0.7)
    test_split_idx = int(n_samples * 0.8)
    
    train_set = (X[:val_split_idx], Y[:val_split_idx])
    val_set = (X[val_split_idx:test_split_idx], Y[val_split_idx:test_split_idx])
    test_set = (X[test_split_idx:], Y[test_split_idx:])
    
    return train_set, val_set, test_set