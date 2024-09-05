import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Step 1: Data Collection
data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000) 
data = data.reshape(-1, 1)

# Step 2: Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 50  # Use the past 50 days to predict the next day
X, y = create_sequences(scaled_data, seq_length)

# Step 3: Model Architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

# Step 4: Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the Model
model.fit(X, y, epochs=5, batch_size=32)

# Step 6: Predicting Future Prices
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the results
plt.plot(data, color='blue', label='Actual Prices')
plt.plot(np.arange(seq_length, len(predicted_prices) + seq_length), predicted_prices, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
