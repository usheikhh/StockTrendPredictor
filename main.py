#NEURAL NETWORK - LSTM MODEL 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

start = '2014-01-01'
end = '2025-12-31'

stock = 'GOOG'

data = yf.download(stock, start, end)

ma_100_days = data.Close.rolling(window=100).mean()
ma_200_days = data.Close.rolling(window=200).mean()
plt.figure(figsize=(12,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close,'g')
plt.show()
