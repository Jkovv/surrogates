import tensorflow as tf
import numpy as np
import random
import os
from core_sta_lstm import STALSTM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_and_eval_sta_lstm(params, train_set, val_set, seed):
    set_seed(seed)
    X_train, Y_train = train_set
    X_val, Y_val = val_set
    
    model = STALSTM(
        params['hidden_size'], 
        Y_train.shape[1:], 
        act_name=params.get('activation', 'ReLU')
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss='mse')
    
    # early-stopping config
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=12, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val),
        epochs=80, 
        batch_size=16, 
        verbose=0, 
        callbacks=[es]
    )
    
    return min(history.history['val_loss']), model.get_weights()