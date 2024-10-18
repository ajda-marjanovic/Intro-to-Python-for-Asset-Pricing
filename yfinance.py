# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:51:43 2023

@author: ajdam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

#%% introduction to yfinance, dictionaries

# let's download prices of Microsoft from 1st January 2020 - 1st September 2023
msft_df = yf.download('MSFT', start='2020-01-01', end='2023-09-01')

msft_df.tail()

# plot adjusted close price
plt.figure(dpi=1080)
plt.figure(figsize=((12,8)))
msft_df['Adj Close'].plot(title='Microsoft adjusted close price',
                          color='steelblue', lw=1.2)

# repeat for Apple
aapl_df = yf.download('AAPL',  start='2020-01-01', end='2023-09-01')

# create a dictionary
stock_data = {'MSFT':msft_df, 'AAPL': aapl_df}

# access an element of a dictionary by its key
stock_data['MSFT']


#%% download prices for multiple stocks from yahoo finance

# create an empty dictionary
stock_data = {}

# all the firms for which we want data
tickers = ['MSFT', 'AAPL', 'TSLA', 'NVDA', 'AMZN']

start_date = '2020-01-01'
end_date = '2023-09-01'

for ticker in tickers:
    stock_df = yf.download(ticker, start_date, end_date)
    stock_data[ticker] = stock_df

#%% plot prices

plt.figure(dpi=1080)
plt.figure(figsize=(12,8))

for stock, stock_df in stock_data.items():
    stock_df['Adj Close'].plot(label=f'{stock} Adj Close')

plt.title('Adjusted close prices for multiple stocks')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()

#%% plot volumes

plt.figure(dpi=1080)
plt.figure(figsize=(12,8))

for stock, stock_df in stock_data.items():
    stock_df['Volume'].plot(label=f'{stock} Volume')

plt.title('Volume for multiple stocks')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()

#%% download SP500

sp500_df = yf.download('SPY', start_date, end_date)

stock_data['SPY'] = sp500_df

#%% calculate average log returns

returns = pd.DataFrame()

for stock, stock_df in stock_data.items():
    returns[stock] = np.log(stock_df['Adj Close']/stock_df['Adj Close'].shift(1)).dropna()

returns.mean()
