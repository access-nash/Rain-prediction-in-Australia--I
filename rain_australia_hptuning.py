# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 07:53:46 2025

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_ra = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Advanced ML Algorithms/Rain_inaus_feature_engineering_final.csv')
df_ra.columns
df_ra.dtypes
df_ra.shape
df_ra.head()
        
missing_values = df_ra.isnull().sum()
print(missing_values)

df_ra['Date'] = pd.to_datetime(df_ra['Date'])
df_ra['Year'] = df_ra['Date'].dt.year
df_ra['Month'] = df_ra['Date'].dt.month
df_ra['Day'] = df_ra['Date'].dt.day
df_ra['Weekday'] = df_ra['Date'].dt.weekday

df_ra = df_ra.drop('Date', axis=1)

categorical_columns = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
df_encoded = pd.get_dummies(df_ra, columns=categorical_columns, drop_first=True)

df_encoded['RainToday'] = df_encoded['RainToday'].map({'No': 0, 'Yes': 1})
df_encoded['RainTomorrow'] = df_encoded['RainTomorrow'].map({'No': 0, 'Yes': 1})
df_encoded.dtypes

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization


X = df_encoded.drop(columns=['RainTomorrow'])
y = df_encoded['RainTomorrow']


# Standardize the features for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression with Lasso (L1) and Ridge (L2) Regularization 
# Lasso
lasso = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
lasso.fit(X_train, y_train)

# Ridge
ridge = LogisticRegression(penalty='l2', solver='saga', max_iter=10000)
ridge.fit(X_train, y_train)

lasso_train_score = accuracy_score(y_train, lasso.predict(X_train))
lasso_test_score = accuracy_score(y_test, lasso.predict(X_test))

ridge_train_score = accuracy_score(y_train, ridge.predict(X_train))
ridge_test_score = accuracy_score(y_test, ridge.predict(X_test))

print(f'Lasso (L1) - Train Accuracy: {lasso_train_score:.4f}, Test Accuracy: {lasso_test_score:.4f}')
print(f'Ridge (L2) - Train Accuracy: {ridge_train_score:.4f}, Test Accuracy: {ridge_test_score:.4f}')

# Hyperparameter Tuning: GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength for Logistic Regression
    'max_iter': [100, 1000, 10000]
}

# GridSearch for Lasso (L1)
grid_search_lasso = GridSearchCV(LogisticRegression(penalty='l1', solver='saga', max_iter=10000), param_grid, cv=5)
grid_search_lasso.fit(X_train, y_train)

# GridSearch for Ridge (L2)
grid_search_ridge = GridSearchCV(LogisticRegression(penalty='l2', solver='saga', max_iter=10000), param_grid, cv=5)
grid_search_ridge.fit(X_train, y_train)

print(f"Best GridSearchCV Parameters for Lasso: {grid_search_lasso.best_params_}")
print(f"Best GridSearchCV Parameters for Ridge: {grid_search_ridge.best_params_}")

# Hyperparameter Tuning: RandomizedSearchCV
param_dist = {
    'C': np.logspace(-4, 4, 20),
    'max_iter': [100, 1000, 10000]
}

# RandomizedSearch for Lasso (L1)
random_search_lasso = RandomizedSearchCV(LogisticRegression(penalty='l1', solver='saga', max_iter=10000), param_dist, n_iter=100, cv=5, random_state=42)
random_search_lasso.fit(X_train, y_train)

# RandomizedSearch for Ridge (L2)
random_search_ridge = RandomizedSearchCV(LogisticRegression(penalty='l2', solver='saga', max_iter=10000), param_dist, n_iter=100, cv=5, random_state=42)
random_search_ridge.fit(X_train, y_train)

print(f"Best RandomizedSearchCV Parameters for Lasso: {random_search_lasso.best_params_}")
print(f"Best RandomizedSearchCV Parameters for Ridge: {random_search_ridge.best_params_}")


# Hyperparameter Tuning: Bayesian Optimization

# Function to optimize for Lasso (L1)
def logistic_regression_lasso_cv(C, max_iter):
    model = LogisticRegression(penalty='l1', C=C, max_iter=int(max_iter), solver='saga')
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Function to optimize for Ridge (L2)
def logistic_regression_ridge_cv(C, max_iter):
    model = LogisticRegression(penalty='l2', C=C, max_iter=int(max_iter), solver='saga')
    return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()

# Set the range of hyperparameters for Bayesian Optimization
pbounds = {'C': (1e-6, 1e+6), 'max_iter': (100, 10000)}

# Bayesian Optimization for Lasso (L1)
optimizer_lasso = BayesianOptimization(
    f=logistic_regression_lasso_cv, 
    pbounds=pbounds, 
    random_state=42
)

optimizer_lasso.maximize(init_points=5, n_iter=50)

# Bayesian Optimization for Ridge (L2)
optimizer_ridge = BayesianOptimization(
    f=logistic_regression_ridge_cv, 
    pbounds=pbounds, 
    random_state=42
)

optimizer_ridge.maximize(init_points=5, n_iter=50)

best_params_lasso = optimizer_lasso.max['params']
best_params_ridge = optimizer_ridge.max['params']

print(f"Best Hyperparameters for Logistic Regression (Lasso): C={best_params_lasso['C']}, max_iter={int(best_params_lasso['max_iter'])}")
print(f"Best Hyperparameters for Logistic Regression (Ridge): C={best_params_ridge['C']}, max_iter={int(best_params_ridge['max_iter'])}")


# Evaluate the best models from each tuning method

# Evaluate GridSearchLasso and GridSearchRidge
print("\nGridSearchLasso - Test Accuracy:", accuracy_score(y_test, grid_search_lasso.best_estimator_.predict(X_test)))
print("GridSearchRidge - Test Accuracy:", accuracy_score(y_test, grid_search_ridge.best_estimator_.predict(X_test)))

# Evaluate RandomizedSearchLasso and RandomizedSearchRidge
print("\nRandomizedSearchLasso - Test Accuracy:", accuracy_score(y_test, random_search_lasso.best_estimator_.predict(X_test)))
print("RandomizedSearchRidge - Test Accuracy:", accuracy_score(y_test, random_search_ridge.best_estimator_.predict(X_test)))

# Evaluate BayesianSearchLasso and BayesianSearchRidge
best_lasso_model = LogisticRegression(penalty='l1', C=best_params_lasso['C'], max_iter=int(best_params_lasso['max_iter']), solver='saga')
best_lasso_model.fit(X_train, y_train)

best_ridge_model = LogisticRegression(penalty='l2', C=best_params_ridge['C'], max_iter=int(best_params_ridge['max_iter']), solver='saga')
best_ridge_model.fit(X_train, y_train)

lasso_test_accuracy = accuracy_score(y_test, best_lasso_model.predict(X_test))
ridge_test_accuracy = accuracy_score(y_test, best_ridge_model.predict(X_test))

print("\nBayesianSearchLasso - Test Accuracy:", lasso_test_accuracy)
print("BayesianSearchRidge - Test Accuracy:", ridge_test_accuracy)


# Evaluate and Plot Confusion Matrices for All Models 
models = [
    ('GridSearch Lasso', grid_search_lasso.best_estimator_),
    ('GridSearch Ridge', grid_search_ridge.best_estimator_),
    ('RandomizedSearch Lasso', random_search_lasso.best_estimator_),
    ('RandomizedSearch Ridge', random_search_ridge.best_estimator_),
    ('BayesianSearch Lasso', best_lasso_model),
    ('BayesianSearch Ridge', best_ridge_model)
]

# Plot confusion matrices for all models
for name, model in models:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Rain', 'Rain'], yticklabels=['No Rain', 'Rain'])
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()