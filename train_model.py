import numpy as np

from keras.models import Model, Sequential
from keras.layers import LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed, ZeroPadding3D

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop


from utilities import *

# list of [46 x 4096]
training_data = prepare_data()

# list of [25 x 240]
y_data = prepare_text_data()
output_data=y_data[0]

def create_model(X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    model = Sequential()

    # Creating encoder network
    model.add(LSTM(hidden_size, input_shape=(X_max_len, 4096)))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model

m=create_model(46,240,25,64,1)
m.summary()

def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences

y_word_to_ix = y_data[1]
y_ix_to_word = y_data[2]

y_sequences = process_data(output_data, 25, y_word_to_ix)

m.load_weights('weights.hdf5')
m.fit(training_data, y_sequences, 1, epochs=200, verbose=1)

m.save_weights('weights.hdf5')
