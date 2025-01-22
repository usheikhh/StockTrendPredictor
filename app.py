import numpy as np
import pandas as pd
import yfinance as yf
import keras.models
import streamlit as st
import matplotlib.pyplot as plt


model = keras.models.load_model("StockPredictionsModel.keras")

st.header('Stock Price Predictor')

stock = st.text_input('Enter Stock Ticker', 'TSLA')
start = '2014-01-01'
end = '2024-12-31'

data = yf.download(stock, start, end)


data_train = pd.DataFrame(data.Close[0:int(len(data)*0.8)]) # 80% of data for training
data_test = pd.DataFrame(data.Close[int(len(data)*0.8):len(data)]) # 20% of data for testing

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) # used to fit the data into the range of 0 to 1

past_100_days = data_train.tail(100)
data_test = pd.concat((past_100_days, data_test), ignore_index=True) # test dataset is ready now 
data_test_scaler = scaler.fit_transform(data_test) # scaling the test data; test data is set now 

x = []
y = []

for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x) # predicting the values of x

scale = 1/scaler.scale_

predict = predict*scale  # changing it back to the normal price values 
y = y*scale


st.subheader('Original vs Predicted Price') 
ma_100_days = data.Close.rolling(window=100).mean()
figure4 = plt.figure(figsize=(12,6))
plt.plot(predict, 'r', label = "Original Price")
plt.plot(y, 'g', label = "Predicted Price")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(figure4)




st.subheader('Price vs Moving Average of 50 Days') 
ma_50_days = data.Close.rolling(window=50).mean()
figure1 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label = "MA 50")
plt.plot(data.Close, 'g', label = 'Original Price')
plt.legend()
plt.show()
st.pyplot(figure1)

st.subheader('Price vs MA 50 vs MA 100') 
ma_100_days = data.Close.rolling(window=100).mean()
figure2 = plt.figure(figsize=(12,6))
plt.plot(ma_50_days, 'r', label = "MA 50")
plt.plot(data.Close, 'g', label = 'Original Price')
plt.plot(ma_100_days, 'b', label = 'MA 100')
plt.legend()
plt.show()
st.pyplot(figure2)

st.subheader('Price vs MA 100 vs MA 200') 
ma_200_days = data.Close.rolling(window=200).mean()
figure3 = plt.figure(figsize=(12,6))
plt.plot(data.Close, 'g', label = 'Original Price')
plt.plot(ma_100_days, 'b', label = 'MA 100')
plt.plot(ma_200_days, 'r', label = 'MA 200')
plt.legend()
plt.show()
st.pyplot(figure3)




