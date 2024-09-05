import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimimzers import Adam
from tensorflow.keras import Sequential 

np.random.seed(42)
time_series_length=1000
time=np.arange(0,time_series_length)
data=np.sin(0.02*time)+np.random.normal(0,0.1,time_series_length)

#anomly
data[500:510]=data[500:510]+3

#plotting
plt.figure(figsize=(12,6))
plt.plot(time,data)
plt.title("Synthetic Time Series Data with Anomalies")
plt.show()

scaler=MinMaxScaler()
data_normalized=scaler.fit_transform(data.reshape(-1,1))
#create sequences using sliding window
def create_sequences(data,window_size):
    sequences=[]
    for i in range(len(data)-window_size):
        sequences.append(data[i:i+window_size])
    return  np.array(sequences)

window_size=50
sequences=create_sequences(data_normalized,window_size)

#split data into training and test set
train_size=int(len(sequences)*0.7)
train_sequences=sequences[:train_size]
test_sequence=sequences[train_size:]

model=Sequential([
    Dense(32,activation='relu',input_shape=(window_size,)),
    Dense(16,activation='relu'),
    Dense(32,activation='relu'),
    Dense(window_size,activation='sigmoid')
])

#compilation of model
model.compile(optimizer=Adam(learning_rate=0.001),lose='mse')
#training 
history=model.fit(train_sequences,train_sequences,
                  epochs=50,
                  batch_size=32,
                  validation_split=0.1,
                  shuffle=True)

reconstructions=model.predict(test_sequence)
mse=np.mean(np.power(test_sequence-reconstructions,2),axis=1)
#plot the reconstruction error
plt.figure(figsze=(12,6))
plt.plot(mse)
plt.title("Reconstruction Error on Test Data")
plt.show()

#set the threshold for anomalies
threshold=np.percentile(mse,95)

#identify anomalies
anomalies=mse>threshold

# Set a threshold for anomalies
threshold = np.percentile(mse, 95)

# Identify anomalies
anomalies = mse > threshold

# Plot the anomalies
plt.figure(figsize=(12, 6))
plt.plot(time[window_size+train_size:], data[window_size+train_size:], label='Data')
plt.plot(time[window_size+train_size:][anomalies], 
         data[window_size+train_size:][anomalies], 
         'ro', label='Anomaly')
plt.title("Detected Anomalies")
plt.legend()
plt.show()

# Set a threshold for anomalies
threshold = np.percentile(mse, 95)

# Identify anomalies
anomalies = mse > threshold

# Plot the anomalies
plt.figure(figsize=(12, 6))
plt.plot(time[window_size+train_size:], data[window_size+train_size:], label='Data')
plt.plot(time[window_size+train_size:][anomalies], 
         data[window_size+train_size:][anomalies], 
         'ro', label='Anomaly')
plt.title("Detected Anomalies")
plt.legend()
plt.show()
