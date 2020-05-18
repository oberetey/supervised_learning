# Lasso selects out the most important for predictiion
# while shrinking the coefficients of certain other features to 0. 
# Its ability to perform feature selection in this way becomes even more useful
# when you are dealing with data involving thousands of features.

# In this script, I will fit a lasso regression to the Gapminder data and plot the coefficients. 
# My hope is the coefficients of some features are shrunk to 0, with only the most important ones remaining.

# The feature and target variable arrays have been loaded as X and y.

# Import Lasso
from sklearn.linear_model import Lasso

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the (gapminder) CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features (fertility) and target (life) variable
y = df['life'].values
X = df['fertility'].values

# Instantiate a lasso regressor: lasso
# Do so with an alpha of 0.4 and specify normalize=True
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients using the coef_ attribute.
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
# Plot the coefficients on the y-axis and column names on the x-axis. 
# This has been done for you, so hit 'Submit Answer' to view the plot!
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
