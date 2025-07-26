from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Flatten, LeakyReLU, SimpleRNN, GRU
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# 1. MinMax Scaling
def minmax_scaling(X, y):
    # X： 2D
    # y:  2D
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    if len(y.shape) == 1:
        y = y.values.reshape(-1, 1)

    # Fit and transform
    scaled_X = X_scaler.fit_transform(X)
    scaled_y = y_scaler.fit_transform(y).flatten()
    
    return scaled_X, scaled_y

# 2. Chronologically Split
def data_split(X, y, num_lags, train_ratio=0.7, val_ratio=0.15):
    """
    X: 2D array-like, shape (n_samples, n_features)
    y: 1D array-like, shape (n_samples,)
    num_lags: int, if 0 then no lagging, just split sequentially
    """
    if num_lags == 0:
        total_len = len(y)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)
        
        x_train = X[:train_end]
        y_train = y[:train_end]
        
        x_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        x_test = X[val_end:]
        y_test = y[val_end:]
        
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    else:
        x_lagged = []
        y_lagged = []

        total_len = len(y)

        for i in range(total_len - num_lags):
            x_lagged.append(X[i:i + num_lags])
            y_lagged.append(y[i + num_lags - 1])
        
        x_lagged = np.array(x_lagged)  # shape: (n_samples, num_lags, n_features)
        y_lagged = np.array(y_lagged)  # shape: (n_samples,)

        n_samples = len(x_lagged)
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        x_train, y_train = x_lagged[:train_end], y_lagged[:train_end]
        x_val, y_val = x_lagged[train_end: val_end], y_lagged[train_end: val_end]
        x_test, y_test = x_lagged[val_end:], y_lagged[val_end:]

        return x_train, y_train, x_val, y_val, x_test, y_test


# callbacks 
def callbacks(model_dir):    
    callbacks = [
        ModelCheckpoint(
            model_dir,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        )
#         ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=5,
#         min_lr=1e-6
#     )
]
    return callbacks

# 3-1.Simple RNN 
def simple_rnn(input_shape, rnn_neurons, dropout, optimizer):
    model = Sequential([
        SimpleRNN(rnn_neurons, 
                  return_sequences=True, 
                  input_shape=input_shape, 
                  dropout=dropout),
        SimpleRNN(rnn_neurons // 2,
                  return_sequences=False),
#         Dense(dense_neurons),  
#         LeakyReLU(alpha=0.01),
        Dense(1)
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error',
        metrics=['mse']
    )
    
    return model

# 3-2 LSTM model
def lstm_model(input_shape, lstm_neurons, dense_neurons, dropout, optimizer):
    model = Sequential([
        LSTM(lstm_neurons, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
#         BatchNormalization(),
        LSTM(lstm_neurons // 2, return_sequences=False),
        Dense(dense_neurons),  
        LeakyReLU(alpha=0.01),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])
    return model

# 3-3 GRU model
def gru_model(input_shape, gru_neurons, dense_neurons, dropout, optimizer):
    model = Sequential([
        GRU(gru_neurons, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
#         BatchNormalization(),
        GRU(gru_neurons // 2, return_sequences=False),
        Dense(dense_neurons),  
        LeakyReLU(alpha=0.01),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])
    return model 

# Train model 
def train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, callbacks_fn, save_dir, verbose=1):
    """
    Train a given Keras model and return the training history.

    Parameters
    ----------
    model : RNN|LSTM|GRU
    x_train : np.ndarray, Training features.
    y_train : np.ndarray, Training targets.
    x_val : np.ndarray, Validation features.
    y_val : np.ndarray, Validation targets.
    batch_size : int, Batch size.
    epochs : int, Number of training epochs.
    callbacks_fn : function, A function that returns a list of callbacks (e.g., callbacks(save_dir)).
    save_dir : str, Directory for saving model checkpoints.
    verbose : int, Verbosity mode (default=1).

    Returns
    -------
    history : Keras History object containing training metrics per epoch.
    """

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        callbacks=callbacks_fn(save_dir),
        verbose=verbose
    )
    
    return history

# Hyperparameter optimization



if __name__ == "__main__":
    df = pd.read_csv('selected_features_and_y.csv')
    df = df.set_index('Date')
    y = df['y1_30_lagged_ma30_pct_change']
    X = df.drop(columns='y1_30_lagged_ma30_pct_change')
    scaled_X, scaled_y = minmax_scaling(X, y)
    print(scaled_X)
    print(scaled_y)
    print(scaled_y.shape)

    # Naive true value
    x_train, y_train, x_val, y_val, x_test, y_test = data_split(scaled_X, scaled_y, num_lags=0)
    df2 = pd.read_csv('feature_engineered.csv')
    ma30_btc_close_pct_change = df2['ma30_btc_close_pct_change']
    print(ma30_btc_close_pct_change)
    y_scaler = MinMaxScaler()
    naive_y_pred = y_scaler.fit_transform(ma30_btc_close_pct_change.to_numpy().reshape(-1, 1))
    print(naive_y_pred)
    y_true = scaled_y
    df = pd.DataFrame({
        'y_true': y_true.flatten(),
        'naive_pred': naive_y_pred.flatten()})
    df.to_csv('naive_and_true.csv')