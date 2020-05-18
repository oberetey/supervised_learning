# in this script i will use the 'fertility' feature of the Gapminder dataset. 
# Since the goal is to predict life expectancy, the target variable here is 'life'. 
# The array for the target variable is set as y and the array for 'fertility' has been set as X_fertility.

# A scatter plot with 'fertility' on the x-axis and 'life' on the y-axis has been generated. 
# As you can see, there is a strongly negative correlation, so a linear regression should be able to capture this trend. 
# my job is to fit a linear regression and then predict the life expectancy: 
# overlaying these predicted values on the plot to generate a regression line.

# i will also compute and print the R2 score using sckit-learn's .score() method.


# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the (gapminder) CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features (fertility) and target (life) variable
y = df['life'].values
X = df['fertility'].values


# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
# Set up the prediction space to range from the minimum to the maximum of X_fertility. 
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
# Fit the regressor to the data (X_fertility and y) 
# and compute its predictions using the .predict() method 
# and the prediction_space array.
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
# to do so Compute and print the R2 score using the .score() method.
print(reg.score(X_fertility, y))

# Plot regression line,
plt.plot(prediction_space, y_pred, color='black', linewidth=3)

# Some where before, the plt.show() fucntion you will provided the plt.scatter() function, 
# Which will result in the Overlay the plot with your linear regression line.
plt.show()
