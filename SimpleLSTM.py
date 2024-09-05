import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#dummy dataa
data=np.array([i for i in range(100)])
labels=np.array([i+1 for i in range(100)])

#reshaping data to be samples,timesteps,features
data=data.reshape((data.shape[0],1,1))
labels=labels.reshape((labels.shape[0],1))

model=Sequential()
model.add(LSTM(50,activation='relu',input_shape=(1,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(data,labels,epochs=5,verbose=1)

#prediction
test_data=np.array([100]).reshape((1,1,1))
prediction=model.predict(test_data)
print(f"Prediction: {prediction}")