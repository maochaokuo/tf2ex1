import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import newaxis
import datetime

trainDataFile = tf.keras.utils.get_file("trainData.csv",
                                        "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                        "-202003/20180702trainData.csv", cache_dir="c:/temp")
dfTd = pd.read_csv(trainDataFile, index_col=None)  # this is when use pandas
npTd0 = dfTd.to_numpy()
npTd = npTd0[:, :, newaxis]
print(dfTd.shape)  # (121293, 32)
print(npTd.shape)  # (121293, 32, 1)

trainTendFile = tf.keras.utils.get_file("trainTend.csv",
                                        "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                        "-202003/20180702trainTend.csv", cache_dir="c:/temp")
dfTt = pd.read_csv(trainTendFile, index_col=None)  # this is when use pandas
npTt = dfTt.to_numpy()
print(dfTt.shape)  # (121293, 1)
print(npTt.shape)  # (121293, 1)

verifyDataFile = tf.keras.utils.get_file("verifyData.csv",
                                         "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                         "-202003/20200302verifyData.csv", cache_dir="c:/temp")

dfVd = pd.read_csv(verifyDataFile, index_col=None)  # this is when use pandas
# print(dfTd.head())
npVd0 = dfVd.to_numpy()
npVd = npVd0[:, :, newaxis]
print(dfVd.shape)  # (6590, 32)
print(npVd.shape)  # (6590, 32, 1)

verifyTendFile = tf.keras.utils.get_file("verifyTend.csv",
                                         "https://raw.githubusercontent.com/maochaokuo/tf2ex1/master/data/201807"
                                         "-202003/20200302verifyTend.csv", cache_dir="c:/temp")
dfVt = pd.read_csv(verifyTendFile, index_col=None)  # this is when use pandas
npVt = dfVt.to_numpy()
print(dfVt.shape)  # (6590, 1)
print(npVt.shape)  # (6590, 1)

filename = 'data/201807-202003/saved_model_50_0.6218512654304504'
model = tf.keras.models.load_model(filename)

# Check its architecture
# print(model.summary())

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictTd = probability_model.predict(npTd)
predictVd = probability_model.predict(npVd)

print(predictTd.shape)
print(predictVd.shape)

print(predictVd[:10])
print(np.argmax(predictVd[:10]))
print(predictTd[:10])
np.savetxt("data/201807-202003/predictTd.csv", predictTd, delimiter=",")
np.savetxt("data/201807-202003/predictVd.csv", predictVd, delimiter=",")
