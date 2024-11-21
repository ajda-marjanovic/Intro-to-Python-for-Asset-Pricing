## Introduction to Python for Asset Pricing / Financial Economics

This repository contains Python files that I have created for a Master's course in **Asset Pricing / Financial Economics**.

The goal is to introduce students to the Python programming language, with a focus on **basic data analysis** and **financial economics applications**.

It is recommended to work through the files in the following order:
1. **introduction.py** (Introduction to Python, Pandas, and NumPy)
2. **yfinance.py** (Yahoo finance package for financial data, Introduction to dictionaries)
3. **regressions.py** (Financial data cleaning, Realized Variance and GARCH, CAPM, Fama-French factor model)
4. **portfolioopt.py** () - 'portfolio.csv' dataset is used in this script
5. **time_series_analysis.py** (Testing for stationarity, Forecasting with ARIMA model) - 'portfolio.csv' dataset is used in this script
6. **regularization.py** (Regularization techniques: Ridge, Lasso) - 'data_ML.csv' dataset is used in this script

     Companion slides for the regularization script can be found [here](https://github.com/ajda-marjanovic/Intro-to-Python-for-Asset-Pricing/blob/d63b2bb1582943fcb03edc54d19621caa971e140/Regularization.pdf)

Two **datasets** are required to work through the scripts in this repository:
- portfolio.csv (This file can be found directly in the repository.)
- data_ML.csv (Due to the its large size (~ 100MB), this file is not included in the repository. It can be downloaded from my Dropbox [here](https://www.dropbox.com/scl/fi/5f1e2ndfif8nlic4uzqp4/data_ML.csv?rlkey=klp8kpwvl0qor9dlag69vw1lq&st=9pu9ifsm&dl=0).)
   
Make sure you have these **libraries** installed:
- pandas
- numpy
- matplotlib
- yfinance
- arch
- pandas_datareader
- statsmodels
- sklearn
