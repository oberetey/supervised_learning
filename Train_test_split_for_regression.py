
# train and test sets are vital to ensure that your supervised learning model is able
# to generalize well to new data. 
# This is true for classification models, and is equally true for linear regression models.

# In this exercise:
# here we will split the Gapminder dataset into training and testing sets, 
# (then fit and predict a linear regression over all features) 
# In addition to computing the R2 score, you will also compute the Root Mean Squared Error (RMSE), 
# (which is another commonly used metric to evaluate regression models.)
# The feature array X and target variable array y have been pre-loaded for you from the DataFrame df.



# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the (gapminder) CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features (fertility) and target (life) variable
y = df['life'].values
X = df['fertility'].values


# Create training and test sets
# Using X and y: 
# (create training and test sets such that 30% is used for testing and 70% for training. 
# Use a random state of 42.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# Create a linear regression regressor called reg_all, 
# fit it to the training set, and evaluate it on the test set.

# First, Create the regressor: reg_all
reg_all = LinearRegression()

# Second, Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Third, Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE

# Compute and print the R2 score using the .score() method on the test set.
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# Compute and print the RMSE. 
# To do this, first compute the Mean Squared Error using the mean_squared_error() function 
# with the arguments y_test and y_pred, 
# and then take its square root using np.sqrt().
print("Root Mean Squared Error: {}".format(rmse))
