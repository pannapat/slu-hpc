#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

# import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras import optimizers
from keras.utils.vis_utils import plot_model

import time
from numba import jit, njit, prange


# In[4]:


def bigrams(words):
    bigrams = []
    for b in words:
        bigrams.append([b[i:i+2] for i in range(len(b)-1)])
    return bigrams


# In[5]:


def prepare(maxlen, dataset_filename='./data/dataset.csv', use_bigram=False):
    # df = pd.read_csv('./data/dataset.csv')
    df = pd.read_csv(dataset_filename)
    X = df['NAME']
    y = df['NATIONALITY']
    num_classes = len(y.unique())

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=69)

    X_tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
              lower=False, char_level=True, oov_token=None)

    y_tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
              lower=True, char_level=False, oov_token=None)

    X_train = X_train_df.values.astype(str) # Otherwise, there's an error when calling 'fit_on_texts' >> AttributeError: 'int' object has no attribute 'lower'
    X_test = X_test_df.values.astype(str) # Otherwise, there's an error when calling 'fit_on_texts' >> AttributeError: 'int' object has no attribute 'lower'

    if use_bigram:
        X_train = bigrams(X_train)

    X_tokenizer.fit_on_texts(X_train)
    X_train = X_tokenizer.texts_to_sequences(X_train)
    X_test = X_tokenizer.texts_to_sequences(X_test)

    X_train = X_tokenizer.sequences_to_matrix(X_train, mode='tfidf')
    X_test = X_tokenizer.sequences_to_matrix(X_test, mode='tfidf')

    # encode from string labels to numerical labels 
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_df.values.astype(str)) # error without astype(str)
    y_test = label_encoder.transform(y_test_df.values.astype(str))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # pad character sequences to have the same length
    X_train = sequence.pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, padding="post", maxlen=maxlen)
    
    max_features = len(X_tokenizer.word_counts)
    
    return [X_train, y_train, X_test, y_test, max_features, num_classes]


# In[6]:


def model(X_train, y_train,
          X_test, y_test,
          max_features,
          maxlen,
          num_classes,
          nn_type='simple_rnn',
          embedding_dims = 50,
          epochs=20,
          batch_size = 23,
         verbose=0):
    
#     print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    if nn_type == 'simple_rnn':
        model.add(SimpleRNN(embedding_dims))
    elif nn_type == 'lstm':
        model.add(LSTM(maxlen))
    
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

#     print(model.summary())
#     print('Train model...')

    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
                verbose=verbose
             )
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size,
                               verbose=verbose)

    # print('Test model score:', score)
    # print('Test model accuracy:', acc)
    return [score, acc]


# In[40]:


MAX_LEN = 30

tuning_list = np.array([
    {    'name': 'simple_rnn',
        'use_bigram': False,
        'maxlen': MAX_LEN,
        'nn_type': 'simple_rnn'
    },
    {
        'name': 'lstm',
        'use_bigram': False,
        'maxlen': MAX_LEN,
        'nn_type': 'lstm'
    },
    {
        'name': 'simple_rnn_with_bigram',
        'use_bigram': True,
        'maxlen': MAX_LEN,
        'nn_type': 'simple_rnn'
    },
    {
        'name': 'lstm_with_bigram',
        'use_bigram': True,
        'maxlen': MAX_LEN,
        'nn_type': 'lstm'
    }
], dtype=np.object)

@jit()
def run():
    for i in prange(tuning_list.shape[0]):
        params = tuning_list[i]
        print('##### {} #####'.format(params['name']))
        [X_train, y_train, X_test, y_test, max_features, num_classes] = prepare(
            maxlen=params['maxlen'], use_bigram=params['use_bigram'])
        [score, acc] = model(nn_type=params['nn_type'],
              X_train=X_train, y_train=y_train, 
              X_test=X_test, y_test=y_test, 
              max_features=max_features, 
              num_classes=num_classes, 
              maxlen=params['maxlen'], 
              verbose=0)
        print('Test model score:', score)
        print('Test model accuracy:', acc)
    return [score, acc]


if __name__ == '__main__':
    begin = time.time()
    [score, accuracy] = run()
    end = time.time()
    elapse = end - begin
    print("Executed in %f secs" % (elapse))

