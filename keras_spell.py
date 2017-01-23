# encoding: utf-8
'''
Created on Nov 26, 2015

@author: tal

Based in part on:
Learn math - https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py

See https://medium.com/@majortal/deep-spelling-9ffef96a24f6#.2c9pu8nlm
"""

Modified by Pavel Surmenok

'''

from __future__ import print_function, division, unicode_literals

import re
import numpy as np
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.layers import recurrent
from keras.models import Sequential
from numpy import zeros as np_zeros # pylint:disable=no-name-in-module
from numpy.random import seed as random_seed

from data import read_news, generate_examples, vectorize

random_seed(123) # Reproducibility

# Parameters for the model and dataset
DATASET_FILENAME = 'data/dataset/news.2011.en.shuffled'
NUMBER_OF_ITERATIONS = 20000
EPOCHS_PER_ITERATION = 5
RNN = recurrent.LSTM
INPUT_LAYERS = 2
OUTPUT_LAYERS = 2
AMOUNT_OF_DROPOUT = 0.3
BATCH_SIZE = 500
HIDDEN_SIZE = 700
INITIALIZATION = "he_normal" # : Gaussian initialization scaled by fan_in (He et al., 2014)
NUMBER_OF_CHARS = 100 # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")


def generate_model(output_len, chars=None):
    """Generate the model"""
    print('Build model...')
    chars = chars or CHARS
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    for layer_number in range(INPUT_LAYERS):
        model.add(recurrent.LSTM(HIDDEN_SIZE, input_shape=(None, len(chars)), init=INITIALIZATION,
                                 return_sequences=layer_number + 1 < INPUT_LAYERS))
        model.add(Dropout(AMOUNT_OF_DROPOUT))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(output_len))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(OUTPUT_LAYERS):
        model.add(recurrent.LSTM(HIDDEN_SIZE, return_sequences=True, init=INITIALIZATION))
        model.add(Dropout(AMOUNT_OF_DROPOUT))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars), init=INITIALIZATION)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class Colors(object):
    """For nicer printouts"""
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def iterate_training(model, X_train, y_train, X_val, y_val, ctable):
    """Iterative Training"""
    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(1, NUMBER_OF_ITERATIONS):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS_PER_ITERATION, validation_data=(X_val, y_val))
        # Select 10 samples from the validation set at random so we can visualize errors
        for _ in range(10):
            ind = random_randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])] # pylint:disable=no-member
            preds = model.predict_classes(rowX, verbose=0)
            q = ctable.decode(rowX[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            if INVERTED:
                print('Q', q[::-1]) # inverted back!
            else:
                print('Q', q)
            print('A', correct)
            print(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)
            print('---')

def main_news():
    """Main"""
    questions, answers = generate_examples(read_news(DATASET_FILENAME))
    chars_answer = set.union(*(set(answer) for answer in answers))
    chars_question = set.union(*(set(question) for question in questions))
    chars = list(set.union(chars_answer, chars_question))
    X_train, X_val, y_train, y_val, y_maxlen, ctable = vectorize(questions, answers, chars)
    print ("y_maxlen, chars", y_maxlen, "".join(chars))
    model = generate_model(y_maxlen, chars)
    iterate_training(model, X_train, y_train, X_val, y_val, ctable)

if __name__ == '__main__':
    main_news()
