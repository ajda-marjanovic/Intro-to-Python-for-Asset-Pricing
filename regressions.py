# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:15:31 2023

@author: ajdam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import arch
import pandas_datareader.data as web
import statsmodels.api as sm

#%% download and prepare financial data
stock_data = {}

tickers = ['MSFT', 'AAPL', 'TSLA', 'NVDA', 'AMZN']

start_date = '2020-01-01'
end_date = '2023-06-01'

for ticker in tickers:
    stock_df = yf.download(ticker, start_date, end_date)
    stock_data[ticker] = stock_df

sp500_df = yf.download('SPY', start_date, end_date)

stock_data['SPY'] = sp500_df

returns = pd.DataFrame()

# calculate log and simple returns, create two columns for each ticker
for stock, stock_df in stock_data.items():
    returns[stock+str('_log')] = np.log(stock_df['Adj Close']/
                                        stock_df['Adj Close'].shift(1)).dropna()
    returns[stock] = stock_df['Adj Close'].pct_change()

# slicing: take every second column starting from the first column (index 0) as log returns
# and every second starting from the second column (index 1) as simple (change) returns
returns_log = returns.iloc[:,::2]
returns_chg = returns.iloc[:,1::2]

# annualize returns and volatility
daily_data = 252 #number of business days in a year
weekly_data = 51
monthly_data = 12

annualized_returns = daily_data*returns_chg.mean()*100
annualized_volatility = np.sqrt(daily_data)*returns_chg.std()*100

# rolling window (monthly average)
returns_chg['SPYm'] = returns_chg['SPY'].rolling(window = 21).mean()
returns_chg[['SPY', 'SPYm']].plot()

#%% variance

# 1) realized variance (noisy with daily data, better to use high-frequency data)
returns_log['RV'] = returns_log['SPY_log']**2

ax = returns_log[['SPY_log', 'RV']].plot(subplots=True,
                                         color= ['g', 'm'],
                                         lw=0.9)

# 2) GARCH(1,1)
model = arch.arch_model(returns_log['SPY_log'], vol = 'Garch', p=1, q=1)

# compare the two volatility models
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
axes[0].plot(returns_log['RV'], label='Realized variance', color='m')
axes[1].plot((model.fit().conditional_volatility)**2,
             label= 'GARCH', color='c')
axes[0].legend()
axes[1].legend()
plt.tight_layout()

#%% regressions

# obtain data from Fama-French library, specifically the 5 daily factors
df = web.DataReader('F-F_Research_Data_Factors_daily',
                    'famafrench')

# merge with our returns dataframe
ff3 = pd.merge(returns_chg, df[0], left_index=True,
               right_index=True, how = 'inner')

msft_df = ff3[['MSFT', 'Mkt-RF', 'SMB', 'HML', 'RF']].copy()

# compute excess return
msft_df['MSFTe'] = msft_df['MSFT']*100 - msft_df['RF']

# Capital Asset Pricing Model (CAPM) - regress excess returns on the market premium

y = msft_df['MSFTe']
x = sm.add_constant(msft_df['Mkt-RF'])

model_CAPM = sm.OLS(y,x).fit()
model_CAPM.summary()

# Fama-French 3-factor model

y = msft_df['MSFTe']
x = sm.add_constant(msft_df[['Mkt-RF', 'SMB', 'HML']])

model_FF3 = sm.OLS(y, x).fit()
model_FF3.summary()

"""
Short commentary:
By looking at the CAPM regression, MSFT has a statistically significant
market beta of 1.14, which means its price is more volatile than the market.
The adjusted R^2 of the FF3 model is higher, implying that the second model fits the data better.
The market beta is similar in magnitude, but MSFT loads negatively on SMB and HML factors,
suggesting that it is a large cap and a growth stock.
"""
