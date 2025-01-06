#NEURAL NETWORK - LSTM MODEL 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

start = '2014-01-01'
end = '2025-12-31'

stock = 'GOOG'

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
data_test = pd.DataFrame(data.Close[int(len(data)*0.8):]) # 20% of data for testing

# print(data_train.shape[0]) # 2216 days for training
# print(data_test.shape[0]) # 554 days for testing

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scaled = scaler.fit_transform(data_train) # Normalizing the training data

x = []
y = []
#taking the first 100 days of data to predict day 101 
for i in range(100, data_train.shape[0]):
    x.append(data_train_scaled[i-100:i]) # rolling window of 100 days
    y.append(data_train_scaled[i,0]) # the next day after 100 days; the target value

