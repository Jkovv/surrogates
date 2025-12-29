import tensorflow as tf
from core_sta_lstm import STALSTM

def train_and_eval_sta_lstm(params, train_set, val_set, seed, grid_size):
    tf.keras.utils.set_random_seed(seed)
    X_train, Y_train = train_set
    X_val, Y_val = val_set

    model = STALSTM(params['hidden_size'], Y_train.shape[1:], grid_size, params['activation'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss='mse')

    b_size = 1 if grid_size >= 250 else (8 if grid_size >= 100 else 32)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=b_size,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    val_loss = model.evaluate(X_val, Y_val, verbose=0)
    return val_loss, model.get_weights()