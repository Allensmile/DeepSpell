# encoding: utf-8

import os
from collections import Counter
import re
import numpy as np
from numpy.random import choice as random_choice, randint as random_randint, shuffle as random_shuffle, seed as random_seed, rand
from numpy import zeros as np_zeros # pylint:disable=no-name-in-module

from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Dropout
from keras.layers import recurrent

# Parameters for the model and dataset
MAX_INPUT_LEN = 40
MIN_INPUT_LEN = 3
INVERTED = True
AMOUNT_OF_NOISE = 0.2 / MAX_INPUT_LEN
NUMBER_OF_CHARS = 100 # 75
CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")

# Some cleanup:
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE) # match all whitespace except newlines
RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(chr(768), chr(769), chr(832),
                                                                                      chr(833), chr(2387), chr(5151),
                                                                                      chr(5152), chr(65344), chr(8242)),
                                  re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
ALLOWED_CURRENCIES = """¥£₪$€฿₨"""
ALLOWED_PUNCTUATION = """-!?/;"'%&<>.()[]{}@#:,|=*"""
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}{}]'.format(re.escape(ALLOWED_CURRENCIES), re.escape(ALLOWED_PUNCTUATION)), re.UNICODE)

# pylint:disable=invalid-name

def add_noise_to_string(a_string, amount_of_noise):
    """Add some artificial spelling mistakes to the string"""
    if rand() < amount_of_noise * len(a_string):
        # Replace a character with a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position + 1:]
    if rand() < amount_of_noise * len(a_string):
        # Delete a character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
    if len(a_string) < MAX_INPUT_LEN and rand() < amount_of_noise * len(a_string):
        # Add a random character
        random_char_position = random_randint(len(a_string))
        a_string = a_string[:random_char_position] + random_choice(CHARS[:-1]) + a_string[random_char_position:]
    if rand() < amount_of_noise * len(a_string):
        # Transpose 2 characters
        random_char_position = random_randint(len(a_string) - 1)
        a_string = (a_string[:random_char_position] + a_string[random_char_position + 1] + a_string[random_char_position] +
                    a_string[random_char_position + 2:])
    return a_string



def vectorize(questions, answers, chars=None):
    """Vectorize the questions and expected answers"""
    print('Vectorization...')
    chars = chars or CHARS
    x_maxlen = max(len(question) for question in questions)
    y_maxlen = max(len(answer) for answer in answers)

    len_of_questions = len(questions)
    ctable = CharacterTable(chars)

    print("X = np_zeros")
    X = np_zeros((len_of_questions, x_maxlen, len(chars)), dtype=np.bool)
    print("X.shape = {}".format(X.shape))

    print("for i, sentence in enumerate(questions):")
    for i in range(len(questions)):
        sentence = questions.pop()
        for j, c in enumerate(sentence):
            X[i, j, ctable.char_indices[c]] = 1
        if (i + 1) % 10000 == 0:
            print("Vectorized {} questions".format(i + 1))

    print("y = np_zeros")
    y = np_zeros((len_of_questions, y_maxlen, len(chars)), dtype=np.bool)
    print("y.shape = {}".format(y.shape))

    print("for i, sentence in enumerate(answers):")
    for i in range(len(answers)):
        sentence = answers.pop()
        for j, c in enumerate(sentence):
            y[i, j, ctable.char_indices[c]] = 1
        if (i + 1) % 10000 == 0:
            print("Vectorized {} answers".format(i + 1))

    # Explicitly set apart 10% for validation data that we never train over
    split_at = int(len(X) - len(X) / 10)
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])

    print(X_train.shape)
    print(y_train.shape)

    return X_train, X_val, y_train, y_val, y_maxlen, ctable


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, maxlen):
        """Encode as one-hot"""
        X = np_zeros((maxlen, len(self.chars)), dtype=np.bool) # pylint:disable=no-member
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        """Decode from one-hot"""
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def clean_text(text):
    """Clean the text - remove unwanted chars, fold punctuation etc."""
    from time import time
    start_time = time()
    result = NORMALIZE_WHITESPACE_REGEX.sub(' ', text.strip())
    print("NORMALIZE_WHITESPACE_REGEX", time() - start_time)
    result = RE_DASH_FILTER.sub('-', result)
    print("RE_DASH_FILTER", time() - start_time)
    result = RE_APOSTROPHE_FILTER.sub("'", result)
    print("RE_APOSTROPHE_FILTER", time() - start_time)
    result = RE_LEFT_PARENTH_FILTER.sub("(", result)
    print("RE_LEFT_PARENTH_FILTER")
    result = RE_RIGHT_PARENTH_FILTER.sub(")", result)
    print("RE_RIGHT_PARENTH_FILTER")
    result = RE_BASIC_CLEANER.sub('', result)
    print("RE_BASIC_CLEANER")
    return result


def read_news(dataset_filename):
    """Read the news corpus"""
    print("reading news")
    news = open(dataset_filename, encoding='utf-8').read()
    print("read news")

    news = clean_text(news)
    print("cleaned text")

    counter = Counter(news)
    most_popular_chars = {key for key, _value in counter.most_common(NUMBER_OF_CHARS)}
    print("Most common chars: " + "".join(sorted(most_popular_chars)).encode('utf-8'))

    lines = [line.strip() for line in news.split('\n')]
    print("Read {} lines of input corpus".format(len(lines)))

    lines = [line for line in lines if line and not bool(set(line) - most_popular_chars)]
    print("Left with {} lines of input corpus".format(len(lines)))

    return lines



def generate_examples(corpus):
    """Generate examples of misspellings"""

    print ("Generating examples")

    questions, answers, seen_answers = [], [], set()

    while corpus:
        line = corpus.pop()
        while len(line) > MIN_INPUT_LEN:
            if len(line) <= MAX_INPUT_LEN:
                answer = line
                line = ""
            else:
                space_location = line.rfind(" ", MIN_INPUT_LEN, MAX_INPUT_LEN - 1)
                if space_location > -1:
                    answer = line[:space_location]
                    line = line[len(answer) + 1:]
                else:
                    space_location = line.rfind(" ") # no limits this time
                    if space_location == -1:
                        break # we are done with this line
                    else:
                        line = line[space_location + 1:]
                        continue
            if answer and answer in seen_answers:
                continue
            seen_answers.add(answer)
            answers.append(answer)
        if random_randint(100000) == 8: # Show some progress
            print('.', end="")

    print('suffle', end=" ")
    random_shuffle(answers)

    print("Done")

    for answer_index, answer in enumerate(answers):
        question = add_noise_to_string(answer, AMOUNT_OF_NOISE)
        question += '.' * (MAX_INPUT_LEN - len(question))
        answer += "." * (MAX_INPUT_LEN - len(answer))
        answers[answer_index] = answer
        assert len(answer) == MAX_INPUT_LEN
        if random_randint(100000) == 8: # Show some progress
            print (len(seen_answers))
            print ("answer:   '{}'".format(answer))
            print ("question: '{}'".format(question))
            print ()
        question = question[::-1] if INVERTED else question
        questions.append(question)

    return questions, answers

def batch_generator(dataset_filename):
    questions, answers = generate_examples(read_news(dataset_filename))
    chars_answer = set.union(*(set(answer) for answer in answers))
    chars_question = set.union(*(set(question) for question in questions))
    chars = list(set.union(chars_answer, chars_question))
    X_train, X_val, y_train, y_val, y_maxlen, ctable = vectorize(questions, answers, chars)
    print ("y_maxlen, chars", y_maxlen, "".join(chars))
