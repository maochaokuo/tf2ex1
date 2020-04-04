import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('GOOG.csv', date_parser = True)
print(data.tail())

data_training = data[data['Date']<'2019-01-01'].copy()
data_test = data[data['Date']>='2019-01-01'].copy()

print(type(data_training))

data_training = data_training.drop(['Date', 'Adj Close'], axis = 1)
data_training0 = data_training.copy()

scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
print(type(data_training))
print(data_training)

# create RNN with 60 timesteps, i.e. look 60 previous time steps

print(data_training[0:10])

X_train = []
y_train = []

for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)

# Building LSTM

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regressior = Sequential()

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 120, activation = 'relu'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))

regressior.summary()

regressior.compile(optimizer='adam', loss = 'mean_squared_error')

#regressior.fit(X_train, y_train, epochs=50, batch_size=32)
regressior.fit(X_train, y_train, epochs=10, batch_size=32)
