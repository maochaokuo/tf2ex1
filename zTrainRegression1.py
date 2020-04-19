import tensorflow as tf
from tensorflow import keras
#
# import pathlib
# import os
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import newaxis
import datetime

startime = datetime.datetime.now()
trainDataFile = tf.keras.utils.get_file("trainData.csv",
                                        "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                        "-201808/20180702trainData.csv", cache_dir="c:/temp")
dfTd = pd.read_csv(trainDataFile, index_col=None)  # this is when use pandas
npTd = dfTd.to_numpy()
# npTd = npTd0[:, :, newaxis]
print(dfTd.shape)  # (6590, 32)
print(npTd.shape)  # (6590, 32, 1)

trainTendPLFile = tf.keras.utils.get_file("trainTendPL.csv",
                                          "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                          "-201808/20180702trainTendPL.csv", cache_dir="c:/temp")
dfTt = pd.read_csv(trainTendPLFile, index_col=None)  # this is when use pandas
npTt = dfTt.to_numpy()
print(dfTt.shape)  # (6590, 1)
print(npTt.shape)  # (6590, 1)

verifyDataFile = tf.keras.utils.get_file("verifyData.csv",
                                         "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                         "-201808/20180801verifyData.csv", cache_dir="c:/temp")

dfVd = pd.read_csv(verifyDataFile, index_col=None)  # this is when use pandas
# print(dfTd.head())
npVd = dfVd.to_numpy()
# npVd = npVd0[:, :, newaxis]
print(dfVd.shape)  # (6891, 32)
print(npVd.shape)  # (6891, 32, 1)

verifyTendPLFile = tf.keras.utils.get_file("verifyTendPL.csv",
                                           "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                           "-201808/20180801verifyTendPL.csv", cache_dir="c:/temp")
dfVt = pd.read_csv(verifyTendPLFile, index_col=None)  # this is when use pandas
npVt = dfVt.to_numpy()
print(dfVt.shape)  # (6891, 1)
print(npVt.shape)  # (6891, 1)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[32]),
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.Flatten(input_shape=(32, 1)),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dropout(0.1),
    # keras.layers.Dense(8, activation='relu'),
    # keras.layers.Dropout(0.05),

    # keras.layers.LSTM(128, activation='relu'),
    # keras.layers.Dropout(0.25),
    # keras.layers.BatchNormalization(),
    # # keras.layers.SimpleRNN(128, activation='relu'),  # faster
    # keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dropout(0.1),

    keras.layers.Dense(1)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(npTd, npTt, epochs=50)  # 10

test_loss, test_acc = model.evaluate(npVd, npVt, verbose=2)
print('\nTest accuracy:', test_acc)

endtime = datetime.datetime.now()
print(startime)
print(endtime)
