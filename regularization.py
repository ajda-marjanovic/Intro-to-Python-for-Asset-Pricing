# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:38:27 2023

@author: amarjanovic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from matplotlib.ticker import ScalarFormatter

#%% download data

df = pd.read_csv('data_ML.csv', index_col=0)
df = df.drop(columns = 'permno')

X = df.iloc[:, 1:]  # Features
y = df.iloc[:, 0] # Target variable (ret)

Xs = StandardScaler().fit_transform(X)

#%% ridge

# in python the tuning parameter is called alpha because lambda is a special function
n_alphas = 100
alphas = np.logspace(2, 7, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(Xs, y)
    coefs.append(ridge.coef_)
    
soln_path = pd.DataFrame(coefs, columns=X.columns, index=alphas)

plt.figure(dpi=1080)
ax = plt.gca()

for col in soln_path.columns:
    ax.plot(soln_path.index, soln_path[col], label=f"{col}")

ax.set_xscale("log")
plt.xlabel("Log lambda")
plt.ylabel("Factor loadings")
plt.title("Ridge coefficients with increasing regularization")
plt.axis("tight")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(soln_path.columns)/6)
plt.show()

top5 = soln_path.iloc[49].abs().sort_values(ascending=False).head(5)
lambda5 = soln_path.index[49]

print(f"Top 5 coefficients for lambda {lambda5} (ridge): \n{top5}")

# top 5: fixed costs to sales, selling and administrative expenses to sales, 
# residual variance, sales to price, variance

#%% choosing the tuning parameter for ridge

cv_scores = []
alphas_cv = np.logspace(2, 7, 20)

for a in alphas_cv:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    scores = cross_val_score(ridge, Xs, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-np.mean(scores))

optimal_alpha = alphas_cv[np.argmin(cv_scores)]

plt.figure(dpi=1080)
plt.plot(alphas_cv, cv_scores, label='Cross-validation MSE')
plt.axvline(optimal_alpha, color='g', linestyle='--', label='Optimal Alpha')
plt.xscale("log")
plt.xlabel("Log lambda")
plt.ylabel("Mean Squared Error")
plt.title("Cross-validation results for Ridge Regression")
plt.legend()
plt.show()

print(f"Optimal lambda (alpha) from cross-validation (ridge): {optimal_alpha}")

#%% lasso

n_alphas_lasso = 100
alphas_lasso = np.logspace(-6, -2, n_alphas_lasso)

coefs_lasso = []
for a in alphas_lasso:
    lasso = linear_model.Lasso(alpha=a, fit_intercept=False)
    lasso.fit(Xs, y)
    coefs_lasso.append(lasso.coef_)
    
soln_path_lasso = pd.DataFrame(coefs_lasso, columns=X.columns, index=alphas_lasso)

fig = plt.figure(dpi=1080)
ax = plt.gca()

for col in soln_path_lasso.columns:
    ax.plot(soln_path_lasso.index, soln_path_lasso[col], label=f"{col}")

ax.set_xscale("log")
plt.xlabel("Log lambda")
plt.ylabel("Factor loadings")
plt.title("Lasso coefficients with increasing regularization")
plt.axis("tight")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(soln_path_lasso.columns)/6)
plt.show()

# fig.savefig('lasso_cv_results.png', transparent=True, bbox_inches='tight')

top5 = soln_path_lasso.iloc[49].abs().sort_values(ascending=False).head(5)
lambda5 = soln_path_lasso.index[49]

print(f"Top 5 coefficients for lambda {lambda5} (lasso): \n{top5}")

#%% choosing the tuning parameter for lasso

cv_scores_lasso = []
alphas_cv_lasso = np.logspace(-6, -2, 20)

for a in alphas_cv_lasso:
    lasso = linear_model.Lasso(alpha=a, fit_intercept=False)
    scores = cross_val_score(lasso, Xs, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores_lasso.append(-np.mean(scores))

optimal_alpha_lasso = alphas_cv_lasso[np.argmin(cv_scores_lasso)]

plt.figure(dpi=1080)
plt.plot(alphas_cv_lasso, cv_scores_lasso, label='Cross-validation MSE')
plt.axvline(optimal_alpha_lasso, color='g', linestyle='--', label='Optimal Alpha')
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
plt.xscale("log")
plt.xlabel("Log lambda")
plt.ylabel("Mean Squared Error")
plt.title("Cross-validation results for Lasso Regression")
plt.legend()
plt.show()

print(f"Optimal lambda (alpha) from cross-validation (lasso): {optimal_alpha_lasso}")

optimal_lasso = linear_model.Lasso(alpha=optimal_alpha_lasso, fit_intercept=False).fit(Xs,y)
selected_features = X.columns[np.where(optimal_lasso.coef_ != 0)[0]]
weights = optimal_lasso.coef_[np.where(optimal_lasso.coef_ != 0)[0]]
abs_weights = np.abs(weights)
lasso_feature_weights = pd.DataFrame({'Feature': selected_features, 'Weight': weights, 'Absolute weights': abs_weights})

