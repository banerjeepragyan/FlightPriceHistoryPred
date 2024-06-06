from flask import Flask, render_template, request, send_file
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import io
from serpapi import GoogleSearch

app = Flask(__name__)

# Load the trained model
model = load_model('../multi_rnn_model.h5')

# Function to predict future prices
def predict_future_prices(data, model, time_step, days):
    predictions = []
    current_data = data[0].tolist()
    
    for _ in range(days):
        input_data = np.array(current_data[-time_step:]).reshape(1, time_step, 1)
        predicted_price = model.predict(input_data)
        current_data.append(predicted_price[0, 0])
        predictions.append(predicted_price[0, 0])
    
    return np.array(predictions)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    dept = request.form['company']
    arr = request.form['ticker']
    date_str = request.form['date']
    
    # For simplicity, let's assume you have a predefined predict.npy file
    # predict_data = np.load('../predict.npy')

    print(dept, arr, date_str)
    params = {
    "engine": "google_flights",
    "departure_id": dept,
    "arrival_id": arr,
    "outbound_date": date_str,
    "currency": "INR",
    "hl": "en",
    "type": "2",
    "api_key": "4ddc2ed2c3e52c83e1ccb052511e154ac18a7722b26e852e2ab446e31af1b6f4" #"8ff19b67f6340c26335822205ded1b109584575455d9af315afaecda9f567fe5"
    }
    predict_data = []
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        price_insights = results["price_insights"]
        price_history = price_insights["price_history"]
        days = len(price_history)
        for k in range(days):
            predict_data.append(price_history[k][1])
    except:
        print("Error calling API")
        return render_template('index.html', plot_url='static/error.png', price="Error calling API")
    predict_data = np.array(predict_data).reshape(1, -1)

    print(predict_data.shape)
    # Normalize the prediction data using MinMaxScaler
    scaler = MinMaxScaler()
    predict_data_normalized = scaler.fit_transform(predict_data)

    # Define the number of days to predict
    d = (datetime.strptime(date_str, "%Y-%m-%d").date() - datetime.today().date()).days

    # Predict the future prices
    future_predictions_normalized = predict_future_prices(predict_data_normalized, model, time_step=10, days=d)

    # Reverse the normalization for predicted data
    padded_array = np.pad(future_predictions_normalized, (0, 61 - future_predictions_normalized.shape[0]), mode='constant', constant_values=1)

    # Reshape the padded array to (1, 61)
    reshaped_array = padded_array.reshape((1, 61))
    print(reshaped_array.shape)
    # Reverse the normalization for predicted data
    future_predictions = scaler.inverse_transform(reshaped_array)

    # Combine past data and future predictions
    all_prices = np.concatenate((predict_data[0], future_predictions.flatten()[:d]))

    # Plot the past 61 days and the next 'd' days
    dates = pd.date_range(start=pd.Timestamp.today() - timedelta(days=60), periods=61 + d, freq='D')
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, all_prices, label='Stock Prices')
    plt.axvline(x=dates[60], color='r', linestyle='--', label='Prediction Start')
    plt.title(f'Ticket Prices: Past 61 Days and Next {d} Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    # plt.savefig(img, format='png')
    plt.savefig('static/plot.png')
    img.seek(0)
    
    # Print the predicted price for the given date
    date = datetime.strptime(date_str, '%Y-%m-%d')
    index = (date - dates[0]).days
    if 0 <= index < len(all_prices):
        predicted_price = all_prices[index]
    else:
        predicted_price = "Date out of range for the available predictions."

    plt.close()

    return render_template('index.html', plot_url='static/plot.png', price="")

# Route to serve the plot image
@app.route('/plot.png')
def plot_png():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)