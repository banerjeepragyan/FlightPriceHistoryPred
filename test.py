import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the testing data from a .npy file
test_data = np.load('test_data.npy')

# Prepare the data for prediction
def create_test_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data)-time_step-23):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[(i+time_step):(i+time_step+23), 0])
    return np.array(X), np.array(y)

time_step = 10
X_test, y_true = create_test_dataset(test_data.reshape(-1, 1), time_step)

# Reshape input to be [samples, time steps, features]
X_test = X_test.reshape(X_test.shape[0], time_step, 1)

# Load the trained model
model = load_model('rnn_model.h5')

# Predict the stock prices for the next 23 days
y_pred = []
for i in range(X_test.shape[0]):
    current_input = X_test[i]
    temp_pred = []
    for _ in range(23):
        current_input = current_input.reshape(1, time_step, 1)
        next_pred = model.predict(current_input, verbose=0)
        temp_pred.append(next_pred[0][0])
        current_input = np.append(current_input[:, 1:, :], [[[next_pred[0][0]]]], axis=1)
    y_pred.append(temp_pred)

# Convert predictions to numpy array
y_pred = np.array(y_pred)

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
