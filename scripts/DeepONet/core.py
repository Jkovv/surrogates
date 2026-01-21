import tensorflow as tf
from tensorflow.keras import layers, Model

class MultiScaleFourierProjection(layers.Layer):
    def __init__(self, num_features=128, scales=[1.0, 10.0], **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scales = scales

    def build(self, input_shape):
        n_per_scale = self.num_features // len(self.scales)
        self.B_list = [self.add_weight(
            name=f"B_{i}", shape=(input_shape[-1], n_per_scale),
            initializer=tf.random_normal_initializer(stddev=s), trainable=False
        ) for i, s in enumerate(self.scales)]

    def call(self, x):
        res = []
        for B in self.B_list:
            p = tf.matmul(x, B)
            res.extend([tf.sin(p), tf.cos(p)])
        return tf.concat(res, axis=-1)

class DeepONetUncertainty(Model):
    def __init__(self, branch, trunk, num_cytokines=6):
        super().__init__()
        self.branch_net, self.trunk_net = branch, trunk
        self.log_vars = tf.Variable(tf.zeros(num_cytokines), trainable=True)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self): return [self.loss_tracker]

    def call(self, inputs):
        b = self.branch_net(inputs[0])
        t = self.trunk_net(inputs[1])
        return tf.nn.softplus(tf.reduce_sum(tf.multiply(b, t), axis=-1))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_p = self(x, training=True)
            mse = tf.reduce_mean(tf.square(y - y_p), axis=0)
            loss = tf.reduce_sum(tf.exp(-self.log_vars) * mse + self.log_vars)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_p = self(x, training=False)
        mse = tf.reduce_mean(tf.square(y - y_p), axis=0)
        loss = tf.reduce_sum(tf.exp(-self.log_vars) * mse + self.log_vars)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

def build_deeponet(params, grid_size, num_cytokines=6):
    act = params.get("activation", "swish")
    lat, tw, nf = params.get("latent_dim", 162), params.get("trunk_width", 215), params.get("n_filters", 32)
    
    bi = layers.Input(shape=(grid_size, grid_size, 12))
    x = layers.Conv2D(nf, (3,3), padding='same', activation=act)(bi)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(nf*2, (3,3), padding='same', activation=act)(x)
    x = layers.GlobalAveragePooling2D()(x)
    bo = layers.Reshape((num_cytokines, lat))(layers.Dense(lat*num_cytokines, activation=act)(x))
    
    ti = layers.Input(shape=(3,))
    y = MultiScaleFourierProjection(num_features=128, scales=[1.0, 10.0])(ti)
    y = layers.Dense(tw, activation=act)(y)
    y = layers.Dropout(0.1)(y) 
    y = layers.Dense(tw, activation=act)(y)
    to = layers.Reshape((num_cytokines, lat))(layers.Dense(lat*num_cytokines, activation=act)(y))
    
    return DeepONetUncertainty(Model(bi, bo), Model(ti, to), num_cytokines)
