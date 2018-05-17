import numpy as np

from keras.models import Model, Sequential
from keras.layers import LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input, MaxPooling3D, TimeDistributed, ZeroPadding3D

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop

from keras.preprocessing.text import text_to_word_sequence
from utilities import *

# list of [46 x 4096]
training_data = prepare_test_data()

# list of sentences [25 x 240]
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

predictions = np.argmax(m.predict(training_data), axis=2)

# Get original sentences
file_name='destest.txt'

f=open(file_name)

sentences=[]
for line in f:
    l=line.split()
    des=" ".join(l[2:])
    if len(des)>0:
        sentences.append(text_to_word_sequence(des))

# convert list of words into sentence
def to_sentence(l):
    s=""
    for w in l:
        s+=w+' '
    return s

# print test results
i=0
j=81
answers=[]
for x in range(0,20):
    answers.append([])
for p in predictions:
    print str(j)+". Expected output: "+to_sentence(sentences[i])
    print str(j)+". Output: ",
    for w in p:
        if w!=0:
            print y_ix_to_word[w],
            answers[i].append(y_ix_to_word[w])
    print ""
    print ""
    i+=1
    j+=

from nltk.translate.bleu_score import modified_precision

# print BLEU scores for output sentences with reference to expected output.
j=81
for i in range(0,20):
    bs = float(modified_precision(sentences[i], answers[i],1))
    print "BLEU Score for video number "+str(j)+": "+str(bs)
    j+=1


