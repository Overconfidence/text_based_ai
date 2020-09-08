from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework']
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples),max_length,dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word) % dimensionality)
        results[i, j, index] = 1

embedding_layer = Embedding(1000, 64)

max_features = 10000
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)