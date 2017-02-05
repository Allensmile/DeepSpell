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

import numpy as np
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.layers import recurrent
from keras.models import Sequential
from numpy.random import seed as random_seed
from numpy.random import randint as random_randint

from data import DataSet

random_seed(123)  # Reproducibility

# Parameters for the model and dataset
DATASET_FILENAME = 'data/dataset/news.2011.en.shuffled'
NUMBER_OF_EPOCHS = 100000
RNN = recurrent.LSTM
INPUT_LAYERS = 2
OUTPUT_LAYERS = 2
AMOUNT_OF_DROPOUT = 0.3
BATCH_SIZE = 500
HIDDEN_SIZE = 700
INITIALIZATION = "he_normal"  # : Gaussian initialization scaled by fan_in (He et al., 2014)
NUMBER_OF_CHARS = 100  # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
INVERTED = True


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


def iterate_training(model, dataset):
    """Iterative Training"""

    X_dev_batch, y_dev_batch = next(dataset.dev_set_batch_generator(1000))

    # Train the model each generation and show predictions against the validation dataset
    for epoch in range(1, NUMBER_OF_EPOCHS):
        print()
        print('-' * 50)
        print('Epoch', epoch)

        for X_batch, y_batch in dataset.train_set_batch_generator(BATCH_SIZE):
            model.fit(X_batch, y_batch, nb_epoch=1, batch_size=BATCH_SIZE)

        # Select 10 samples from the dev set at random so we can visualize errors
        for _ in range(10):
            ind = random_randint(0, len(X_dev_batch))
            row_X, row_y = X_dev_batch[np.array([ind])], y_dev_batch[np.array([ind])]
            preds = model.predict_classes(row_X, verbose=0)
            q = dataset.character_table.decode(row_X[0])
            correct = dataset.character_table.decode(row_y[0])
            guess = dataset.character_table.decode(preds[0], calc_argmax=False)

            if INVERTED:
                print('Q', q[::-1])  # inverted back!
            else:
                print('Q', q)

            print('A', correct)
            print(Colors.ok + '☑' + Colors.close if correct == guess else Colors.fail + '☒' + Colors.close, guess)
            print('---')


def main_news():
    """Main"""
    dataset = DataSet(DATASET_FILENAME)
    model = generate_model(dataset.y_max_length, dataset.chars)
    iterate_training(model, dataset)

if __name__ == '__main__':
    main_news()
