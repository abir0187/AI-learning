# Example: Predict Price of House, given #Rooms
# Need: Regression Algorithm, predicts _continous values_ (like price)
#
# * Uses Example Data available in Sklearn (Boston Housing Data)
#
# Task: "I want 5 rooms, how much does a house cost in Boston?"
#
# Steps:
# 1. Load Data
# 2. Algorithm is trained on data.
# 3. Algorithm predicts price, given #Rooms
#
# ~> Watch the video to see further explanation.

# Load sklearn functionality
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

# 1. Load Data
boston = datasets.load_boston()
X = boston.data[:, np.newaxis, 5]  # only 5th column, # rooms
Y = boston.target # price

# Split train/test set
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.50)

# 2. Algorithm is trained on data.
lm = LinearRegression()
lm.fit(X_train, Y_train)

# 3. Algorithm predicts price, given #Rooms
Y_pred = lm.predict(X_test)

# Create a plot showing predictions
plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred, color='red', linewidth=3)

plt.xlabel("Number of Rooms")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices")
plt.show()

# Make some prediction
print( lm.predict([[8]]) )
print( lm.predict([[5]]) )
