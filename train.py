import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, Dropout, GRU, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# Load the training data from a .npy file
train_data = np.load('train.npy')

# Normalize the data
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data)

# Prepare the data for RNN
def create_dataset(data, time_step=1):
    X, Y = [], []
    for arr in data:
        for i in range(len(arr)-time_step-1):
            a = arr[i:(i+time_step)]
            X.append(a)
            Y.append(arr[i + time_step])
    return np.array(X), np.array(Y)

time_step = 3
X_train, y_train = create_dataset(train_data_normalized, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(-1, time_step, 1)

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with validation split and save it
checkpoint = ModelCheckpoint('flight_model.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, epochs=150, batch_size=256, verbose=2, validation_split=0.2, callbacks=[checkpoint], shuffle=True)

print("Model trained and saved as 'flight_model.h5'")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
