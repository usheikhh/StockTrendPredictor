#NEURAL NETWORK - LSTM MODEL 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

start = '2014-01-01'
end = '2025-12-31'

stock = 'AAPL' # Tesla stock data

data = yf.download(stock, start, end) # yahoo finance data download 
data.dropna(inplace=True)  

ma_100_days = data.Close.rolling(window=100).mean() # 100 days moving average
ma_200_days = data.Close.rolling(window=200).mean() # 200 days moving average
# plt.figure(figsize=(12,6))
# plt.plot(ma_100_days, 'r')
# plt.plot(ma_200_days, 'b')

# plt.plot(data.Close,'g')
# plt.show()

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.8)]) # 80% of data for training
data_test = pd.DataFrame(data.Close[int(len(data)*0.8):len(data)]) # 20% of data for testing

# print(data_train.shape[0]) # 2216 days for training
# print(data_test.shape[0]) # 554 days for testing

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0,1)) # used to fit the data into the range of 0 to 1
data_train_scaled = scaler.fit_transform(data_train) # Normalizing the training data

x = []
y = []
#taking the first 100 days of data to predict day 101 
for i in range(100, data_train.shape[0]):
    x.append(data_train_scaled[i-100:i]) # rolling window of 100 days
    y.append(data_train_scaled[i,0]) # the next day after 100 days; the target value


x,y = np.array(x), np.array(y) # converting the lists into numpy arrays



# Model Creation 
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

# all layer will be placed in a sequential manner 
# output of one layer would be the input of the next layer


# Sequential will be used to predict the data based on the time series 
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = ((x.shape[1],1)))) 
    # 50 neurons in the layer
    # 4 layers for the LSTM Sequential model; each layer with different number of neurons 
    # activation function is relu; used to calculate RNN 
    # return_sequences = True; output of one layer is the input to another layer
model.add(Dropout(0.2))
    # dropout layer to prevent overfitting; 20% of the neurons will be dropped out

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))   
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu')) # no other layer after this, so retrun_sequences = False
model.add(Dropout(0.5))

model.add(Dense(units = 1)) # Dense unit is the output layer; 1 neuron in the output layer


model.compile(optimizer= 'adam', loss = 'mean_squared_error')   

model.fit(x,y,epochs= 50, batch_size= 32, verbose=1) # fitting the model with the training data01

# model.summary()

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index= True) # concatenating the past 100 days with the test data
data_test_scaled = scaler.fit_transform(data_test) # Normalizing the test data

x = []
y = []
#taking the first 100 days of data to predict day 101 
for i in range(100, data_test.shape[0]):
    x.append(data_test_scaled[i-100:i]) # rolling window of 100 days
    y.append(data_test_scaled[i,0]) # the next day after 100 days; the target value

x,y = np.array(x), np.array(y) # converting the lists into numpy arrays

y_predict = model.predict(x) # predicting the values for the test data

scale = 1/scaler.scale_ # creating the scale to scale the predicted values

y_predict = y_predict*scale # scaling the predicted values
y = y*scale # scaling the actual values

plt.figure(figsize=(12,6))
plt.plot(y_predict, 'r', label = 'Predicted Stock Price')
plt.plot(y, 'g', label = 'Actual Stock Price')
plt.xlabel('Time in days')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()

model.save('Stock Predictions Model.keras') # saving the model for future use




