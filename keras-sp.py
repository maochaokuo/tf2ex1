import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
#from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
#from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
import datetime
import sys

print(datetime.datetime.now())  # about 14 seconds

SEQ_LEN = 20  # how long of a preceeding sequence to collect for RNN
#SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


print(datetime.datetime.now())  # about 14 seconds

main_df = pd.DataFrame()  # begin empty

# ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
ratios = ["LTC-USD"]
for ratio in ratios:  # begin iteration

    ratio = ratio.split('.csv')[0]  # split away the ticker from the file-name
    print(ratio)

    url = f'https://raw.githubusercontent.com/maochaokuo/tensorflow1/master/sharedfiles/{ratio}.csv'
    dataset = url  # f'training_datas/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df) == 0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

print(datetime.datetime.now())  # about 14 seconds

print(main_df.head(10))

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

print(main_df.head(15))  # how did we do??

main_df.dropna(inplace=True)

## here, split away some slice of the future data from the main main_df.
times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

print(main_df.head(10))


def preprocess_df(df): # df: input dataframe
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.
    print(df.head())
    print(df.shape)
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
    l=0
    for i in df.values:  # iterate over the values, each i become a sub array to be appended, this is to convert from dataframe to ndarray
        # i is df row, without index (time here)
        prev_days.append([n for n in i[:-1]])  # store all but the target
        '''
        if l<5:
            print([np.array(prev_days), i[-1]])
        '''
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!
        l=l+1
    #print(prev_days) very painful, deque cannot print anything but all
    #print(len(sequential_data))
    random.shuffle(sequential_data)  # shuffle for good measure.
    '''
    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    '''
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!
    #return np.array(X), np.array(y)

validation_x, validation_y = preprocess_df(validation_main_df)
print(len(validation_x[0]))
print(validation_x[0][0])
print(validation_y[0])

train_x, train_y = preprocess_df(main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

print(train_x.shape)
print(train_x.shape[1:])
train_x[0][0]

print(datetime.datetime.now())  # about 4 minutes
model = Sequential()

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True)) # 128 hard to change
model.add(Dropout(0.25)) # 0.25 best
model.add(BatchNormalization()) # this is needed

model.add(LSTM(128)) # 128 hard to change
model.add(Dropout(0.2)) # 0.2 best

model.add(Dense(32, activation='relu')) # 32 hard to change
model.add(Dropout(0.2)) # 0.2 best

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
'''
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("./{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
'''
print('before train model')
print(type(train_x))
print(train_x.shape)
train_y=np.array(train_y)
print(type(train_y))
print(train_y.shape)
print(type(validation_x))
validation_y=np.array(validation_y)
print(type(validation_y))

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    #callbacks=[tensorboard, checkpoint],
)

print('after train model')
# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("./{}".format(NAME))

print(datetime.datetime.now())  # about 4 minutes