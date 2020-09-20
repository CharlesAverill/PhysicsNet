import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Attention, ELU, Dropout
from utils import *
from sklearn.preprocessing import scale

import tensorflow.compat.v1 as v1

config = v1.ConfigProto()
config.gpu_options.allow_growth = True
session = v1.InteractiveSession(config=config)


def bidirectional_lstm(input_shape=(636, 128)):
    model = Sequential()

    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
    model.add(Attention(636))
    model.add(Dropout(0.2))
    model.add(Dense(400))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mse',
                  optimizer='adam')

    return model


df = pd.read_csv("./data/scores.csv", index_col=0)

X = np.array(convert_wav_to_image(df))
X = np.array(normalize_dataset(X))
Y = df["score"].values
Y = scale(Y)

"""
plt.figure(figsize=(15,10))
plt.title('Visualization of audio file', weight='bold')
plt.imshow(X[1])
plt.show()
"""

bdlstm = bidirectional_lstm(X[0].shape)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=.1, patience=15)
hist = bdlstm.fit(X,
                  Y,
                  batch_size=256,
                  epochs=100,
                  validation_split=.3,
                  callbacks=[es]
                  )

loss = hist.history['loss']
val_loss = hist.history['val_loss']
stopped_epoch = es.stopped_epoch
epochs = range(stopped_epoch + 1)

plt.figure(figsize=(15, 5))
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(['Training loss', 'Validation loss'], fontsize=16)
plt.show()
