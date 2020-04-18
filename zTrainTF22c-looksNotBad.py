import tensorflow as tf
from tensorflow import keras
#
# import pathlib
# import os
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import newaxis

trainDataFile = tf.keras.utils.get_file("trainData.csv",
                                        "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                        "-201808/20180702trainData.csv", cache_dir="c:/temp")
dfTd = pd.read_csv(trainDataFile, index_col=None)  # this is when use pandas
# print(dfTd.head())
npTd0 = dfTd.to_numpy()
npTd = npTd0[:, :, newaxis]
# print(dfTd.shape)  # (6590, 32)
# print(type(npTd))  # <class 'numpy.ndarray'>
print(npTd.shape)  # (6590, 32)

trainTendFile = tf.keras.utils.get_file("trainTend.csv",
                                        "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                        "-201808/20180702trainTend.csv", cache_dir="c:/temp")
dfTt = pd.read_csv(trainTendFile, index_col=None)  # this is when use pandas
# print(dfTt.head())
npTt = dfTt.to_numpy()
# npTt = npTt.astype(float)
print(dfTt.shape)  # (6590, 1)
print(npTt.shape)  # (6590, 1)

verifyDataFile = tf.keras.utils.get_file("verifyData.csv",
                                         "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                         "-201808/20180801verifyData.csv", cache_dir="c:/temp")

dfVd = pd.read_csv(verifyDataFile, index_col=None)  # this is when use pandas
# print(dfTd.head())
npVd0 = dfVd.to_numpy()
npVd = npVd0[:, :, newaxis]
print(dfVd.shape)  # (6891, 32)
print(npVd.shape)  # (6891, 32, 1)

verifyTendFile = tf.keras.utils.get_file("verifyTend.csv",
                                         "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                         "-201808/20180801verifyTend.csv", cache_dir="c:/temp")
dfVt = pd.read_csv(verifyTendFile, index_col=None)  # this is when use pandas
# print(dfTt.head())
npVt = dfVt.to_numpy()
print(dfVt.shape)  # (6891, 1)
print(npVt.shape)  # (6891, 1)

model = keras.Sequential([
    keras.layers.LSTM(128, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),
    # keras.layers.LSTM(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),

    # # keras.layers.LSTM(128, return_sequences=True),
    #
    # # keras.layers.Flatten(input_shape=(32, 1)),
    # keras.layers.Dense(128, activation='relu'),
    #
    # keras.layers.Dropout(0.2),
    # # keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    # # keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dropout(0.1),
    # keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(npTd, npTt, epochs=300)

test_loss, test_acc = model.evaluate(npVd, npVt, verbose=2)
print('\nTest accuracy:', test_acc)

# trainData_slices = tf.data.Dataset.from_tensor_slices(dict(df))
# print(tf.data.experimental.cardinality(trainData_slices))
# print('feature_batch')
# for feature_batch in trainData_slices.take(1):
#   for key, value in feature_batch.items():
#     print("  {!r:20s}: {}".format(key, value))

# trainData_batches = tf.data.experimental.make_csv_dataset(
#     trainDataFile, batch_size=5, label_name="sectionMinuteNth")
# print('trainDataCSV')
# for feature_batch, label_batch in trainData_batches.take(1):
#   print("'sectionMinuteNth': {}".format(label_batch))
#   print("features:")
#   for key, value in feature_batch.items():
#     print("  {!r:20s}: {}".format(key, value))
