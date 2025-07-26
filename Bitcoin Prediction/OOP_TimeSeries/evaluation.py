from modeling import data_split, minmax_scaling
from tensorflow.keras.models import load_model
import pandas as pd








if __name__ == "__main__":
    

    rnn_best_model = load_model(r"D:/DCU/practicum/2025-mcm-cairo/src/timeseries_oop/best_rnn.keras")
    print(rnn_best_model.summary())
    import zipfile
    print(zipfile.is_zipfile(r"D:\DCU\practicum\2025-mcm-cairo\src\timeseries_oop\best_rnn.keras"))
    # df = pd.read_csv('selected_features_and_y.csv')
    # df = df.set_index('Date')
    # y = df['y1_30_lagged_ma30_pct_change']
    # X = df.drop(columns='y1_30_lagged_ma30_pct_change')
    # scaled_X, scaled_y = minmax_scaling(X, y)
    # x_train, y_train, x_val, y_val, x_test, y_test = data_split(X, y, num_lags)
