# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:10:36 2023

@author: WIN
"""

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Load the diabetes dataset
diabetes = load_diabetes()

# Preprocess the data
X = StandardScaler().fit_transform(diabetes.data)
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an MLP model
mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000)

# Train the model
mlp.fit(X_train, y_train)

# Evaluate the model
score = mlp.score(X_test, y_test)
print("R^2 score:", score)

# R^2 score: 0.46150686623228765, hidden_layer_sizes=(16, 8)
# R^2 score: 0.4380976292446448, hidden_layer_sizes=(16, 8)
# R^2 score: 0.47201385144961695, hidden_layer_sizes=(100, 50)
# R^2 score: 0.48616233328458436, hidden_layer_sizes=(100, 50)