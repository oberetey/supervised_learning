# i will now demonstatrate evaluating a model with tuned hyperparameters on a hold-out set. 
# The feature array and target variable array from the diabetes dataset are loaded as X and y.

# In addition to C, logistic regression has a 'penalty' hyperparameter 
# (which specifies whether to use 'l1' or 'l2' regularization.) 
# my job in this exercise is to create a hold-out set, 
# tune the 'C' and 'penalty' hyperparameters of a logistic regression classifier,
# (using GridSearchCV on the training set.)


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the (gapminder) CSV file into a DataFrame: df
df = pd.read_csv('diabetes.csv')

X = df.iloc[:, 0:8]
y = df.iloc[:, 8:9]

# Create the hyperparameter grid
# Use the array c_space as the grid of values for 'C'.
c_space = np.logspace(-5, 8, 15)
# For 'penalty', specify a list consisting of 'l1' and 'l2'.
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
# Use a test_size of 0.4 and random_state of 42. 
# In practice, the test set here will function as the hold-out set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Tune the hyperparameters on the training set using GridSearchCV with 5-folds. 
# This involves first instantiating the GridSearchCV object with 
# (the correct parameters and then fitting it to the training data.)
# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
# Print the best parameter and best score obtained from 
# (GridSearchCV by accessing the best_params_ and best_score_ attributes of logreg_cv.)
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
