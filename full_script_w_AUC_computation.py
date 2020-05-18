# Say i have a binary classifier that in fact is just randomly making guesses. 
# It would be correct approximately 50% of the time, and the resulting ROC curve would be a diagonal line
# (in which the True Positive Rate and False Positive Rate are always equal.) 
# The Area under this ROC curve would be 0.5. 
# This is one way in which the AUC, is an informative metric to evaluate a model. 
# If the AUC is greater than 0.5, the model is better than random guessing. 
#Always a good sign!

# Here i will calculate AUC scores using the roc_auc_score() function 
# (from sklearn.metrics as well as by performing cross-validation on the diabetes dataset.)

# X and y, along with training and test sets X_train, X_test, y_train, y_test, have been pre-loaded for you, 
# (and a logistic regression classifier logreg has been fit to the training data.)


# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score 

# Import necessary modules
from sklearn.metrics import roc_curve
#the dataset is for PIMA Indians diabetes

# The feature and target variable arrays X and y have been pre-loaded, 
from sklearn.model_selection import train_test_split

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the (gapminder) CSV file into a DataFrame: df
df = pd.read_csv('diabetes.csv')

X = df.iloc[:, 0:8]
y = df.iloc[:, 8:9]

# Create training and test sets with 40% (or 0.4) of the data used for testing. Use a random state of 42. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
# Instantiate a LogisticRegression classifier called logreg.
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
# Using the logreg classifier, which has been fit to the training data, 
# compute the predicted probabilities of the labels of the test set X_test. 
# Save the result as y_pred_prob.
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
# Compute the AUC scores by performing 5-fold cross-validation. 
# Use the cross_val_score() function and specify the scoring parameter to be 'roc_auc'.
cv_auc = cross_val_score(logreg, X, y,cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
