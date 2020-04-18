import tensorflow as tf
#
# import pathlib
# import os
# import matplotlib.pyplot as plt
import pandas as pd

# import numpy as np

trainDataFile = tf.keras.utils.get_file("trainData.csv",
                                        "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                        "-201808/20180702trainData.csv")
dfTd = pd.read_csv(trainDataFile, index_col=None)  # this is when use pandas
print(dfTd.head())

trainTendFile = tf.keras.utils.get_file("trainTend.csv",
                                        "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                        "-201808/20180702trainTend.csv")
dfTt = pd.read_csv(trainTendFile, index_col=None)  # this is when use pandas
print(dfTt.head())

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
