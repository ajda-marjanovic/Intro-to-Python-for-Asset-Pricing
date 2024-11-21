# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:16:45 2023

@author: amarjanovic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier
from pypfopt import plotting

#%% import data

df = pd.read_csv('portfolio.csv', index_col=0)

#%% plot normalized prices (start of period = 100), SP500 in bold

normalized = (df/df.iloc[0])*100

plt.figure(dpi=1080)
plt.plot(normalized, lw= 0.7)
plt.plot(normalized['SPY'], lw = 2, c='black')
xtick_loc = normalized.index[::126]
plt.xticks(xtick_loc,rotation=45)

# worst and best performers of the sample
bottom3 = normalized.iloc[-1].sort_values(ascending=True).iloc[:3]
top3 = normalized.iloc[-1].sort_values(ascending=False).iloc[:3]

#%% compute returns and variance, SP500 is our benchmark

returns = df.pct_change().dropna()

benchmark = returns['SPY']
assets = returns.drop(columns='SPY')

benchRet = 252*benchmark.mean()
assetRet = 252*assets.mean()
benchVol = np.sqrt(252)*benchmark.std()
assetVol = np.sqrt(252)*assets.std()
cov_matrix = 252 * assets.cov()

# risk-return plot

plt.figure(dpi=1080)
plt.scatter(assetVol, assetRet)
plt.scatter(benchVol, benchRet)
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.annotate('S&P', (benchVol, benchRet), fontsize=8, ha='left', va = 'bottom')
for i, asset_name in enumerate(assets):
    plt.annotate(asset_name, (assetVol[i], assetRet[i]), fontsize=8, ha='left', va = 'bottom')

#%% correlation matrix

plt.figure(dpi=1080)
sns.heatmap(assets.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Notice that there is a cluster of tech stocks with high correlation (AAPL, ADBE, AMZN, META, MSFT)
# and a cluster of consumer staples stocks with high correlation (KO, PEP, PG)

#%% use package to find the minimum volatility portfolio (portfolio with the lowest risk)

# define expected returns and covariance matrix
mu = assetRet
S = cov_matrix

ef = EfficientFrontier(mu, S)

# specify which portfolio we want
ef.min_volatility()

# clean_weights rounds weights and clips near zeros
weights = ef.clean_weights()

# we will get expected annual return,volatility and Sharpe ratio printed in the console
ef.portfolio_performance(verbose=True)

# let us plot the weights as a pie chart

pd.Series(weights).plot.pie(figsize=(10,10))

#%% keep minimum volatility portfolio with short-selling

mu = assetRet
S = cov_matrix

# allow short-selling by setting weight bounds to (-1, 1) i.e. weights can also be negative
ef2 = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

ef2.min_volatility()

weights2 = ef2.clean_weights()

ef2.portfolio_performance(verbose=True)

# Allowing short-selling relaxes the constraints on the optimization problem,
# expanding the feasible set. This increases the likelihood of finding a portfolio
# with lower volatility. Indeed, this portfolio has lower volatility compared to
# the one obtained in the previous example.

#%% minimum volatility portfolio with weight constraints

mu = assetRet
S = cov_matrix

ef3 = EfficientFrontier(mu, S)

# constraint: no more than 15% of portfolio invested in a single stock
ef3.add_constraint(lambda w: w <= 0.15)

ef3.min_volatility()

weights3 = ef3.clean_weights()

ef3.portfolio_performance(verbose=True)

pd.Series(weights3).plot.pie(figsize=(10,10))

# This restriction reduces the feasible set, leading to a less optimal solution.
# Indeed, the volatility of this portfolio is higher than in the first example
# without restrictions. Keep in mind that the objective of these examples was finding
# the portfolio with the lowest possible volatility.

#%% maximum Sharpe ratio portfolio with 15% allocation constraint

mu = assetRet
S = cov_matrix

ef4 = EfficientFrontier(mu, S)

ef4.add_constraint(lambda w: w <= 0.15)

ef4.max_sharpe()

weights4 = ef4.clean_weights()

ef4.portfolio_performance(verbose=True)

pd.Series(weights4).plot.pie(figsize=(10,10))

#%% efficient rontier plot: set of portfolios that have the highest expected return
# for a given a value of risk (volatility)

# generate 100000 portfolios with random weights
mus = []
stds = []
sharpes = []
for _ in range(100000):
    w = np.random.dirichlet(np.ones(len(mu)))
    ret = mu.dot(w)
    std = np.sqrt(w.dot(S @ w))
    mus.append(ret)
    stds.append(std)
    sharpes.append(ret / std)


# plot efficient frontier using the package
ef = EfficientFrontier(mu, S)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# populate the plot with previously generated random portfolios
scatter = ax.scatter(stds, mus, marker=".", c=sharpes, cmap="plasma")

# add color bar based on Sharpe ratio values
cbar = plt.colorbar(scatter)
cbar.set_label('Sharpe Ratios')

ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.show()
