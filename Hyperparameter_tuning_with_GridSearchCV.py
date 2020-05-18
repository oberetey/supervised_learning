# how to tune the n_neighbors parameter of the KNeighborsClassifier() 
# (using GridSearchCV on the voting dataset.)
# i will now do so using logistic regression on the diabetes dataset instead!

# Like the alpha parameter of lasso and ridge regularization, 
# logistic regression also has a regularization parameter: C.

# What is parameter C:
# (C controls the inverse of the regularization strength, 
# and this is what you will tune in this exercise.)
# A large C can lead to an overfit model, while a small C can lead to an underfit model.

# The hyperparameter space for C has been setup for you. 
# my job is to use GridSearchCV and logistic regression to find:
# (the optimal C in this hyperparameter space.) 
# The feature array is available as X and target variable array is available as y.

# You may be wondering why you aren't asked to split the data into training and test sets.
# Good observation! 
# Here, we want you to focus on the process of setting up the hyperparameter grid
# and performing grid-search cross-validation. 
# In practice, you will indeed want to hold out a portion of your data for evaluation purposes,
# and you will learn all about this in the next video!


from sklearn.model_selection import cross_val_score 

# Import necessary modules
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV 

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the (gapminder) CSV file into a DataFrame: df
df = pd.read_csv('diabetes.csv')

X = df.iloc[:, 0:8]
y = df.iloc[:, 8:9]

# Setup the hyperparameter grid by using c_space as the grid of values to tune C over.
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

#Use GridSearchCV with 5-fold cross-validation to tune C:

# Instantiate the GridSearchCV object: logreg_cv
# (Inside GridSearchCV(), specify the classifier, parameter grid, and number of folds to use.)
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
# (Use the .fit() method on the GridSearchCV object to fit it to the data X and y.)
logreg_cv.fit(X, y)

# Print the tuned parameters and score
# Print the best parameter and best score obtained from GridSearchCV 
# (by accessing the best_params_ and best_score_ attributes of logreg_cv.)
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))
