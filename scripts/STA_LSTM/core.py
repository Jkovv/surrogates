import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import random
import numpy as np

def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention_map = self.conv(concat)
        return inputs * attention_map

class STALSTMUncertainty(Model):
    def __init__(self, grid_size, num_cytokines=6, n_filters=64, hidden_dim=256):
        super().__init__()
        self.grid_size = grid_size
        self.num_cytokines = num_cytokines
        
        self.encoder = tf.keras.Sequential([
            layers.TimeDistributed(layers.Conv2D(n_filters, (3,3), padding='same', activation='swish')),
            layers.TimeDistributed(layers.BatchNormalization()),
            layers.TimeDistributed(SpatialAttention()),
            layers.TimeDistributed(layers.MaxPooling2D((2,2))), 
            layers.TimeDistributed(layers.Conv2D(n_filters*2, (3,3), padding='same', activation='swish')),
            layers.TimeDistributed(layers.GlobalAveragePooling2D())
        ])

        self.lstm = layers.LSTM(hidden_dim, return_sequences=False)

        self.decoder = tf.keras.Sequential([
            layers.Dense(10 * 10 * 64, activation='swish'),
            layers.Reshape((10, 10, 64)),
            layers.Resizing(self.grid_size, self.grid_size),
            layers.Conv2DTranspose(64, (3,3), padding='same', activation='swish'),
            layers.Conv2D(num_cytokines, (1,1), activation='softplus') 
        ])

        self.log_vars = tf.Variable(tf.zeros(num_cytokines), trainable=True)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.lstm(x)
        return self.decoder(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_p = self(x, training=True)
            mse = tf.reduce_mean(tf.square(y - y_p), axis=(0, 1, 2))
            loss = tf.reduce_sum(tf.exp(-self.log_vars) * mse + self.log_vars)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_p = self(x, training=False)
        mse = tf.reduce_mean(tf.square(y - y_p), axis=(0, 1, 2))
        loss = tf.reduce_sum(tf.exp(-self.log_vars) * mse + self.log_vars)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
