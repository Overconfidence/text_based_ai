from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Embedding, Dense, SimpleRNN
from keras.layers import LSTM
from keras.models import Sequential
import matplotlib as plt
import numpy as np

# Listing 6.21 Numpy implementation of a simple RNN
timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.concatenate(successive_outputs, axis=0)
# end of Listing 6.21 Numpy implementation of a simple RNN

# Listing 6.22 Preparing the IMDB data
max_features = 1000
maxlen = 500
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)
# end of Listing 6.22 Preparing the IMDB data

# Listing 6.27 Using the LSTM layer in Keras
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
#end of Listing 6.27 Using the LSTM layer in Keras

'''
# Listing 6.23 Training the model with Embedding and SimpleRNN layers
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# end of Listing 6.23 Training the model with Embedding and SimpleRNN layers
'''
# Listing 6.24 Plotting results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', labels='Training loss')
plt.plot(epochs, val_loss, 'b', labels='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# end of Listing 6.24 Plotting results
# Book: 6.3 (page 207)