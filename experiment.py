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


class Attention(tf.keras.layers.Layer):

    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        from tensorflow.keras import backend as K

        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)


def bidirectional_lstm(input_shape=(636, 128)):
    model = Sequential()

    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
    model.add(Attention(636))
    model.add(Dropout(0.2))
    model.add(Dense(400))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


df = pd.read_csv("./data/scores.csv", index_col=0)

X = np.array(convert_wav_to_image(df))
X = np.array(normalize_dataset(X))
Y = df["score"].values
Y = scale(Y)

bdlstm = bidirectional_lstm(X[0].shape)

es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
hist = bdlstm.fit(X,
                  Y,
                  batch_size=256,
                  epochs=10,
                  validation_split=.2,
                  callbacks=[es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
stopped_epoch = es.stopped_epoch
epochs = range(stopped_epoch + 1)

plt.figure(figsize=(15, 5))
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss over epochs', weight='bold', fontsize=22)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend(['Training loss', 'Validation loss'], fontsize=16)
plt.show()
