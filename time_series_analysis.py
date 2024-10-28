# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:04:34 2023

@author: amarjanovic
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

#%% download and visualize Coca-Cola price for the period 2017-2022

df = pd.read_csv('portfolio.csv', index_col=0)

stock = df['KO']
stock.index = pd.to_datetime(stock.index)

plt.figure(dpi=1080)
plt.plot(stock, lw = 1.5, c='black')
plt.xticks(stock.index[::124].strftime('%Y-%m'), rotation=-60)
plt.ylabel('Coca-Cola price')

"""
In this exercise session we will use autoregressive moving-average (ARMA) models for forecasting.
Put simply, the variable is modelled as a series of lags of the same variable (AR) and
a series of lags of the error terms (MA) + the innovation at time t.

ARIMA models are used when the series is non-stationary. The 'I' stands for integrated,
which you can think of in this terms: If a time-series is integrated of order 1 denoted I(1), 
taking first-order difference of the variable will make the time-series stationary."""

#%% test stationarity of prices (augmented Dickey-Fuller test) -> ARMA or ARIMA

# null hypothesis: unit root (non-stationary)
adf_test = adfuller(stock, autolag='AIC')
adf_output = pd.Series(adf_test[0:3], index=['Test Statistic', 'p-Value', 'Lags'])

# we cannot reject: prices are non-stationary, we should use ARIMA models

# check returns (here we reject the null, we assume returns to be stationary)
adf_test_ret = adfuller(stock.pct_change().dropna(), autolag='AIC')
adf_output_ret = pd.Series(adf_test_ret[0:3], index=['Test Statistic', 'p-Value', 'Lags'])

#%% fit ARIMA models on the train set and compute mean square error on the test set

df = pd.DataFrame({'Date': stock.index, 'KO': stock.values})
df['Date'] = pd.to_datetime(df['Date'])

split = 1200

train_data = df[:split]
test_data = df[split:]

results_list = []

for p in range(5): # choose a suitable range for p
    for q in range(5): # choose a suitable range for q
        try:
            model = ARIMA(train_data.iloc[:,1], order=(p, 1, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(len(test_data))
            errors = test_data.iloc[:,1] - forecast
            mse = (errors ** 2).mean()
            results_list.append({"p": p, "q": q, "MSE": mse})
        except:
            continue

results = pd.DataFrame(results_list, columns=["p", "q", "MSE"])

# select the parametrization with the smallest mean square error
results.loc[results['MSE'].idxmin()]

# usually the best performing model will not have too many lags ->
# more lags i.e. more complex models fit the data better,
# but we have the curse of dimensionality, complex models are more difficult to
# estimate precisely because they have too many parameters

#%% plot forecasts using ARIMA(2,1,2)

model = ARIMA(train_data.iloc[:,1], order=(2, 1, 3))
forecasts = model.fit().get_forecast(len(test_data)).summary_frame()

new_df = pd.merge(test_data, forecasts, left_index=True, right_index=True, how='left')

plt.figure(dpi=1080)
plt.plot(train_data['Date'], train_data['KO'], label='Train Data', lw=1.5, c='blue')
plt.plot(new_df['Date'], new_df['KO'], label='Actual', lw=1.5, c='k')
plt.plot(new_df['Date'], new_df['mean'], label='Forecasted', lw=1.5, c='red')
plt.fill_between(new_df['Date'], new_df['mean_ci_lower'], new_df['mean_ci_upper'], color='k', alpha=0.05, label='95% Confidence Interval')
plt.xticks(train_data['Date'][::126].dt.strftime('%Y-%m-%d'), rotation=-60)
plt.legend()
plt.show()

#%% forecast during COVID

# locate peak (start the forecast 5 days before peak)
peak = df[df['Date'] < '2020-06-01']['KO'].idxmax()-5

train_data = df.iloc[:peak]
test_data = df.iloc[peak:(peak+59)]

model = ARIMA(train_data.iloc[:,1], order=(2, 1, 3))
forecasts = model.fit().get_forecast(len(test_data)).summary_frame()

new_df = pd.merge(test_data, forecasts, left_index=True, right_index=True, how='left')

plt.figure(dpi=1080)
plt.plot(train_data['Date'], train_data['KO'], label='Train Data', lw=1.5, c='blue')
plt.plot(new_df['Date'], new_df['KO'], label='Actual', lw=1.5, c='k')
plt.plot(new_df['Date'], new_df['mean'], label='Forecasted', lw=1.5, c='red')
plt.fill_between(new_df['Date'], new_df['mean_ci_lower'], new_df['mean_ci_upper'], color='k', alpha=0.10, label='95% Confidence Interval')
plt.xticks(train_data['Date'][::126].dt.strftime('%Y-%m-%d'), rotation=-60)
plt.legend()
plt.show()

#%% rolling forecasts (one-step ahead) during COVID

forecasted_data = []

for i, (_, row) in enumerate(test_data.iterrows()):
    model = ARIMA(train_data['KO'], order=(2, 1, 2)).fit()
    forecast = model.get_forecast(1).summary_frame()

    forecasted_data.append({
        'Date': row['Date'],
        'KO': row['KO'],
        'mean': forecast.iloc[0,0],
        'mean_ci_lower': forecast.iloc[0,2],
        'mean_ci_upper': forecast.iloc[0,3]})
    
    train_data = train_data.append(row)

forecasted_data = pd.DataFrame(forecasted_data, columns=['Date', 'KO', 'mean', 'mean_ci_lower', 'mean_ci_upper'])

plt.figure(dpi=1080)
plt.plot(forecasted_data['Date'], forecasted_data['KO'], label='Actual', lw=1.5, c='blue')
plt.plot(forecasted_data['Date'], forecasted_data['mean'], label='Forecasted', lw=1.5, c='red')
plt.fill_between(forecasted_data['Date'], forecasted_data['mean_ci_lower'],
                 forecasted_data['mean_ci_upper'], color='k', alpha=0.10, label='95% Confidence Interval')
plt.xticks(forecasted_data['Date'][::10], forecasted_data['Date'][::10].dt.strftime('%Y-%m-%d'), rotation=-60)
plt.legend()
plt.show()