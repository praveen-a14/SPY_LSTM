import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout

# Download data from yfinance, store in csv to avoid downloading on every run

data = yf.download("SPY", start="2014-01-01", end="2023-12-31", group_by="ticker")
data.to_csv('spy.csv')
data_2024 = yf.download("SPY", start="2024-01-01", end="2024-12-31", group_by="ticker")
data_2024.to_csv('spy_2024.csv') 

# Get 2014-23 and 2024 $SPY data, adjust for weird csv
spy = pd.read_csv('spy.csv', skiprows=[0,2], parse_dates=['Price'])
spy_2024 = pd.read_csv('spy_2024.csv', skiprows=[0,2], parse_dates=['Price'])
spy.rename(columns={'Price': 'Date'}, inplace=True)
spy_2024.rename(columns={'Price': 'Date'}, inplace=True)
prices = spy['Adj Close'].values.reshape(-1, 1)
prices_2024 = spy_2024['Adj Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)  # Fit scaler only on historical data
prices2024_scaled = scaler.transform(prices_2024)

# Split the data into training and testing sets
train_size = int(len(prices_scaled) * 0.7)
train_data, test_data = prices_scaled[:train_size], prices_scaled[train_size:]

# sequence creating function
def create_sequences(data, sequence_length=7):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Create training and testing sequences
sequence_length = 7
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Get the last sequence from 2023 data and combine with 2024
last_days_2023 = prices_scaled[-sequence_length:]
prices_2024_with_2023 = np.concatenate((last_days_2023, prices2024_scaled), axis=0)
X_2024, _ = create_sequences(prices_2024_with_2023)

#Create LSTM Model
model = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=7, batch_size=32)
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Train Loss: {train_loss:.6f}')
print(f'Test Loss: {test_loss:.6f}')

# Predict closing prices
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
pred_2024 = model.predict(X_2024)

# Denormalize the predictions
train_predictions = scaler.inverse_transform(train_predictions)
y_train_unscaled = scaler.inverse_transform(y_train)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_unscaled = scaler.inverse_transform(y_test)
predictions_2024 = scaler.inverse_transform(pred_2024)

#Line up dates, make sure the start of 2024 uses the last days of 2023
train_dates = spy['Date'][sequence_length:train_size + sequence_length]  
test_dates = spy['Date'][train_size + sequence_length: len(spy) - sequence_length] 
dates_2024 = spy_2024['Date']
dates_2023_2024 = np.concatenate([spy['Date'][-sequence_length:], dates_2024])

# Remove timezone and convert back to pandas series
train_dates = train_dates.dt.tz_localize(None)
test_dates = test_dates.dt.tz_localize(None)
dates_2023_2024 = pd.Series(dates_2023_2024)
dates_2024 = pd.Series(dates_2024)
dates_2023_2024 = dates_2023_2024.dt.tz_localize(None)
dates_2024 = dates_2024.dt.tz_localize(None)
full_dates = np.concatenate([train_dates, test_dates, dates_2024])


last_days_2023 = spy['Adj Close'][-sequence_length:]
spy_2024_prices = spy_2024['Adj Close']
spy_2024_prices = np.concatenate([last_days_2023, spy_2024_prices])
spy_2024_prices = spy_2024_prices.reshape(-1, 1)
predictions_2024 = predictions_2024[:len(dates_2023_2024)]

actual_prices = np.concatenate([y_train_unscaled, y_test_unscaled, spy_2024_prices])[sequence_length:].flatten()
predicted_prices = np.concatenate([train_predictions, test_predictions, predictions_2024])[sequence_length:].flatten()

current_date = pd.to_datetime("now").tz_localize(None) 
spy_2024['Date'] = spy_2024['Date'].dt.tz_localize(None) 
current_date_index = (spy_2024['Date'] - current_date).abs().idxmin()

last_available_data = prices2024_scaled[current_date_index - sequence_length: current_date_index]
rolling_sequence = last_available_data

# Calculate how many days into the future to predict (business days till the end of 2024)
future_days = len(pd.date_range(current_date, '2024-12-31', freq='B'))

# Empty list to store predictions
rolling_predictions = []

# Predict future days using the rolling forecast method
for _ in range(future_days):
    prediction = model.predict(rolling_sequence.reshape(1, sequence_length, 1))
    rolling_predictions.append(prediction[0, 0])
    rolling_sequence = np.append(rolling_sequence[1:], prediction, axis=0)

last_actual_date = spy_2024['Date'].iloc[-1]
predicted_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), end='2024-12-31', freq='B')
# Denormalize the rolling predictions
rolling_predictions = scaler.inverse_transform(np.array(rolling_predictions).reshape(-1, 1))
rolling_predictions = rolling_predictions.flatten()

plt.figure(figsize=(14, 7))
plt.plot(full_dates, actual_prices, label='Actual Prices')
plt.plot(full_dates[sequence_length:], predicted_prices, label='2014-2023 Seen Predictions')
plt.plot(dates_2024, predictions_2024, label='2024 Unseen Predicitons')
plt.plot(predicted_dates, rolling_predictions, label='2024 Rest of Year Predicted Prices')
plt.title('$SPY Adjusted Close Price Prediction with Rolling Forecast (2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
