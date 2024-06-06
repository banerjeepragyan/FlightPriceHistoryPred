import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the testing data from a .npy file
test_data = np.load('test.npy')

org_shape = test_data.shape

# Normalize the test data using the same scaler used for training data
scaler = MinMaxScaler()
test_data_normalized = scaler.fit_transform(test_data)

# Prepare the data for prediction
def create_test_dataset(data, time_step):
    X = []
    for arr in data:
        for i in range(len(arr)-time_step):
            a = arr[i:(i+time_step)]
            X.append(a)
    return np.array(X)

# Plot the predicted prices
def plot_predictions(test_data, predictions, time_step):
    for i in range(test_data.shape[0]):
        plt.figure(figsize=(10, 6))
        actual_prices = test_data[i, time_step:]
        predicted_prices = predictions[i, :len(actual_prices)]
        # dates = pd.date_range(start=pd.Timestamp.today(), periods=len(actual_prices), freq='D')
        dates = np.arange(1, len(actual_prices)+1)
        plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
        plt.plot(dates, predicted_prices, label='Predicted Prices', color='red')
        plt.title(f'Company {i+1} Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

time_step = 10
X_test = create_test_dataset(test_data_normalized, time_step)

# Reshape input to be [samples, time steps, features]
X_test = X_test.reshape(-1, time_step, 1)

# Load the trained model
model = load_model('multi_rnn_model.h5')

# Predict the stock prices for the next day for each company
predictions = model.predict(X_test)

# Reshape predictions to match the original shape of the test data
predictions_reshaped = predictions.reshape((org_shape[0], org_shape[1]-time_step))
print(predictions_reshaped.shape)

# Calculate the amount of padding required
padding_width = [(0, 0) for _ in range(len(org_shape))]
padding_width[-1] = (0, org_shape[-1] - predictions_reshaped.shape[-1])

# Pad the array with ones
arr_padded = np.pad(predictions_reshaped, padding_width, mode='constant', constant_values=1)

# Reverse the normalization for predicted data
predictions_original_scale = scaler.inverse_transform(arr_padded)

# Print the reshaped predictions
print("Predictions for each flight:")
print(predictions_original_scale[:, time_step:])

# Calculate overall RMSE
rmse = np.sqrt(mean_squared_error(test_data[:, time_step:], predictions_original_scale[:, time_step:]))
print("Overall RMSE:", rmse)

plot_predictions(test_data, predictions_original_scale, time_step)