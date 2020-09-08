from keras.preprocessing.text import Tokenizer
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
