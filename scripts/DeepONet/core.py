import tensorflow as tf
from tensorflow.keras import layers, Model

def build_deeponet(params, grid_size, num_cytokines=6):
    def get_p(name):
        if hasattr(params, 'suggest_int'):
            if name == "activation": return params.suggest_categorical(name, ["gelu", "swish", "tanh"])
            if name == "n_filters": return params.suggest_categorical(name, [32, 64])
            return params.suggest_int(name, 128, 256)
        return params[name]

    n_filters, latent_dim = get_p("n_filters"), get_p("latent_dim")
    trunk_width, activation = get_p("trunk_width"), get_p("activation")
    init = tf.keras.initializers.GlorotUniform()

    branch_input = layers.Input(shape=(grid_size, grid_size, 12))
    x = layers.Conv2D(n_filters, (3, 3), padding='same', activation=activation, kernel_initializer=init)(branch_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(latent_dim * num_cytokines, activation=activation)(x)
    x = layers.LayerNormalization()(x) 
    branch_reshaped = layers.Reshape((num_cytokines, latent_dim))(x)

    trunk_input = layers.Input(shape=(3,))
    y = layers.Dense(trunk_width, activation=activation, kernel_initializer=init)(trunk_input)
    y = layers.LayerNormalization()(y)
    y = layers.Dense(trunk_width, activation=activation, kernel_initializer=init)(y)
    y = layers.LayerNormalization()(y)
    y = layers.Dense(latent_dim * num_cytokines, activation=activation)(y)
    y = layers.LayerNormalization()(y)
    trunk_reshaped = layers.Reshape((num_cytokines, latent_dim))(y)

    dot = layers.Multiply()([branch_reshaped, trunk_reshaped])
    merged = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(dot)
    merged = layers.Lambda(lambda x: x * 0.1)(merged)
    final_output = layers.Activation('sigmoid')(merged)
    return Model(inputs=[branch_input, trunk_input], outputs=final_output)
