# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:17:37 2023

@author: WIN
"""

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense

# Load data
diabetes = load_diabetes()
# Preprocess the data
X = StandardScaler().fit_transform(diabetes.data)
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Build model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='mse', optimizer='adam')

# Train model
model.fit(X_train, y_train, epochs=200, batch_size=10, validation_split=0.2)

# Evaluate model
y_pred = model.predict(X_test)  #, y_test)
score = r2_score(y_test, y_pred)
print("R^2 score:", score)
# R^2 score: 0.4121845510590546
# R^2 score: 0.4787058293145606
# R^2 score: 0.5015287710964408