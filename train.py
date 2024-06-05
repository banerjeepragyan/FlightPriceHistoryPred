import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the training data from a .npy file
data = np.load('dummy_train.npy')
train_data_1, train_data_2 = data

# Concatenate the two arrays to create the training dataset
train_data = np.concatenate((train_data_1, train_data_2))

# Prepare the data for RNN
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X_train, y_train = create_dataset(train_data.reshape(-1, 1), time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], time_step, 1)

# Create the RNN model
model = Sequential()
model.add(SimpleRNN(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model and save it
checkpoint = ModelCheckpoint('rnn_model.h5', save_best_only=True, monitor='loss', mode='min')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2, callbacks=[checkpoint])

print("Model trained and saved as 'rnn_model.h5'")
