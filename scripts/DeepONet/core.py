import tensorflow as tf
from tensorflow.keras import layers, Model

class DeepONetUncertainty(Model):
    def __init__(self, branch, trunk, num_cytokines=6):
        super().__init__()
        self.branch_net = branch
        self.trunk_net = trunk
        
        self.log_vars = tf.Variable(tf.zeros(num_cytokines), trainable=True, name="loss_scales")
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs):
        x_branch, x_trunk = inputs
        b = self.branch_net(x_branch)
        t = self.trunk_net(x_trunk)
        
        dot = tf.multiply(b, t)
        merged = tf.reduce_sum(dot, axis=-1)
        
        return tf.nn.softplus(merged)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            
            mse_per_channel = tf.reduce_mean(tf.square(y - y_pred), axis=0)
            
            precision = tf.exp(-self.log_vars)
            loss = tf.reduce_sum(precision * mse_per_channel + self.log_vars)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        mse_per_channel = tf.reduce_mean(tf.square(y - y_pred), axis=0)
        precision = tf.exp(-self.log_vars)
        loss = tf.reduce_sum(precision * mse_per_channel + self.log_vars)
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

def build_deeponet(params, grid_size, num_cytokines=6):
    def get_p(name):
        if hasattr(params, 'suggest_int'):
            if name == "activation": return params.suggest_categorical(name, ["gelu", "swish"])
            if name == "n_filters": return params.suggest_categorical(name, [32, 64])
            if name == "latent_dim": return params.suggest_int(name, 128, 256)
            if name == "trunk_width": return params.suggest_int(name, 128, 256)
        return params[name]

    n_filters = get_p("n_filters")
    latent_dim = get_p("latent_dim")
    trunk_width = get_p("trunk_width")
    activation = get_p("activation")
    init = tf.keras.initializers.HeNormal()

    branch_input = layers.Input(shape=(grid_size, grid_size, 12), name="branch_input")
    x = layers.Conv2D(n_filters, (3, 3), padding='same', activation=activation, kernel_initializer=init)(branch_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(n_filters * 2, (3, 3), padding='same', activation=activation, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(latent_dim * num_cytokines, activation=activation, kernel_initializer=init)(x)
    branch_output = layers.Reshape((num_cytokines, latent_dim))(x)
    branch_model = Model(branch_input, branch_output, name="Branch_CNN")

    trunk_input = layers.Input(shape=(3,), name="trunk_input")
    y = layers.Dense(trunk_width, activation=activation, kernel_initializer=init)(trunk_input)
    y = layers.LayerNormalization()(y)
    y = layers.Dense(trunk_width, activation=activation, kernel_initializer=init)(y)
    y = layers.LayerNormalization()(y)
    y = layers.Dense(latent_dim * num_cytokines, activation=activation, kernel_initializer=init)(y)
    y = layers.LayerNormalization()(y)
    trunk_output = layers.Reshape((num_cytokines, latent_dim))(y)
    trunk_model = Model(trunk_input, trunk_output, name="Trunk_MLP")

    return DeepONetUncertainty(branch_model, trunk_model, num_cytokines)
