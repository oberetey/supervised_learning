
# Classification reports and confusion matrices are great methods to quantitatively evaluate model performance,
#( while ROC curves provide a way to visually evaluate models.) 
# most classifiers in scikit-learn have a .predict_proba() method 
# (which returns the probability of a given sample being in a particular class.) 
# Having built a logistic regression model, i will now evaluate its performance by plotting an ROC curve.
# standing for Receiver Operator Characteristic 
# In doing so, you'll make use of the .predict_proba() method and become familiar with its functionality.

# Here, i will be working with the PIMA Indians diabetes dataset. 
# i start by setting up The classifier and fit it to the training data and is available as logreg.

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

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
# This has been done for you, so hit 'Submit Answer' to see how logistic regression compares to k-NN!
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute predicted probabilities: .predict_proba() method 
# (which returns the probability of a given sample being in a particular class. y_pred_prob)
# Using the logreg classifier, which has been fit to the training data, 
# (compute the predicted probabilities of the labels of the test set X_test.) 
# Save the result as y_pred_prob.
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
# Use the roc_curve() function with y_test and y_pred_prob 
# (and unpack the result into the variables fpr, tpr, and thresholds.)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot the ROC curve with fpr on the x-axis and tpr on the y-axis.
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# Understing logistic regression.
# Despite its name, logistic regression is used in classification problems, not regression problems.
# Given one feature, log reg will output a probability, p, with respect to the target variable.
# If p is greater than 0 (point) 5, we label the data as ‘1’; if p less than 0 (point) 5, we label it ‘0’.
# this is only one threshold value
# visually  the placement of the multiple variables in a graph is split by a stright line, that is usally draw at a angle
# this is where the binary nature of the roc is best made obvious.
# this line is refered to as a linear decision boundary, The range of 0 to 1.0,
#indicates the capacity percentage (%) of the models ability to perform class separation


# Understanding metrics logistic regression metrics:

# True positive Rate (TPR) uses the same function as the Recall, 
# It judges how much is the proportion TP out of TP and FP combined.
# Rate of correct positive

# Specificity is the amount True Negatives (TN out of TN +FP)
# It gives perspective to the correct negatives predicted,
# through their relationship to wrongly predicted positives.

# False Positive Rate (FPR) ) substact the Specificity function
# from 1, to give an inverse value of True negative which Specificity focuses on
# (proportiate to False Positive  in the Specificity formula)
