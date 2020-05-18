
# In Chapter 1, you evaluated the performance of your k-NN classifier based on its accuracy. 
# being accuracy is not always an informative metric, can support it more deeply by evaluating the performance
# of binary classifiers by computing a confusion matrix and generating a classification report.

# classification report usally come with a support column. 
# The support gives the number of samples of the true response that lie in that class for Example, 
# if the support was the number of Republicans or Democrats in the test set on which the classification report was computed. 
# The precision, recall, and f1-score columns, then, gave the respective metrics for that particular class.

# Here, i will work with the PIMA Indians dataset obtained from the UCI Machine Learning Repository. 
# The goal is to predict whether or not a given female patient will contract diabetes
# based on features such as BMI, age, and number of pregnancies. 
# Therefore, it is a binary classification problem. 
# A target value of 0 indicates that the patient does not have diabetes,
# while a value of 1 indicates that the patient does have diabetes. 
#As in Chapters 1 and 2, the dataset has been preprocessed to deal with missing values.

# The dataset has been loaded into a DataFrame df and the feature and target variable arrays X and y have been created from the DataFRame using iloc. 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# here i will  train a k-NN classifier to the data and evaluate its performance
# by generating a confusion matrix and classification report.

# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the (gapminder) CSV file into a DataFrame: df
df = pd.read_csv('diabetes.csv')

X = df.iloc[:, 0:8]
y = df.iloc[:, 8:9]

# Create training and test set
# Create training and testing sets with 40% of the data used for testing. Use a random state of 42.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate a k-NN classifier with 6 neighbors,
# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
# Compute and print the confusion matrix and classification report using the confusion_matrix() and classification_report() functions.
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Understanding a confusion matrix - Given any model, we can fill in the confusion matrix according to its predictions.
# True Positive - In the top left square, we have the number of spam emails correctly labeled;
# True Negative - In the bottom right square, we have the number of real emails correctly label;
# False Negative In the top right, the number of spams incorrectly labeled;
# False Positive in the bottom left, the number of real emails incorrectly labeled.


# Additoanl examples of above: 
# Correctly labeled spam emails are referred to as true positives and correctly labeled real emails as true negatives.
# While incorrectly labeled spam will be referred to as false negatives and incorrectly labeled real emails as false positives.

# Errors: distinguish between the types of False

# Type 1 error
# False Positive, is an error made when you predict a positive but it’s actually false meaning it is not a positive,
# your prediction was wrong.
# Type 2 error
# False Negative, is an error made when you predict a Negative but it’s actually false meaning it is not a Negative,
# your prediction was wrong.

# Usually, the “class of interest” is called the positive class.
# As we are trying to detect spam, this makes spam the positive class.
# Lastly, Which class you call positive is really up to you.



# UNderstanding  Classiftication report
# So why do we care about the confusion matrix?

# First, note that you can retrieve accuracy from the confusion matrix:
# it’s the sum of the diagonal divided by the total sum of the matrix.
# once, this division is done There are several other important metrics, that you can  calculate from the confusion matrix.

# Precision, which is the number of true positives divided by the total number of true positives and false positives.
# This is the amount predicted of True positives out of the True positives and False positives.
# it is Presented as Percentage, It is also called the positive predictive value or PPV.
# For example, this could  the number of correctly labeled spam emails divided by
# the total number of emails divided by the total number of emails classified as spam

# Recall, which is the number of true positives and false negatives.
# This is the ability of prediction of True positives out of the True positives and False negatives.
# it is Presented as Percentage
# this is also called sensitivity, hit rate, or true positive rate.

# Accuracy is tricky it would seem that recall could serve as such but, instead accuracy is:
# Out of all classes how much of the classification were correct.
# is is Presented as Percentage

# The F1-score is defined as two time the product of the precision and recall divided by the sum of the precision and recall,
# in other words, it’s the harmonic mean of precision and recall.
# A harmonic mean, a mean function, that won’t be distorted by changes of original values into larger number in an attempt to drastically change the outcome result as longer as one of the original values.
# To put it in plain language, high precision means that our classifier had a low false positive rate,
# that is, not many real emails were predicted as being spam.
# Intuitively, high recall means that our classifier predicted most positive or spam emails correctly.







