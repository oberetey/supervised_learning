# The Gapminder dataset that i worked with in previous scripts,
# also contained a categorical 'Region' feature, 
# which i dropped in previous exercises since you did not have the tools to deal with it. 


# my job in this exercise is to explore the values in a specific feature. 
# Boxplots are particularly useful for visualizing categorical features such as this.


# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Use pandas to create a boxplot showing the variation of:
# life expectancy ('life') by region ('Region'). 
# To do so, pass the column names in to df.boxplot() (in that order).

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()
