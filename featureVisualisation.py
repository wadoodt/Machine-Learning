# Do not forget to run this cell to import libraries,otherwise a lot of code in the workbook will not work!
import numpy as np # A useful package for dealing with mathematical processes, it can deal with multi-dimensional arrays such as matrices intuitively.
import pandas as pd #a common package for viewing tabular data
import seaborn as sns
import sklearn.linear_model, sklearn.datasets 
from sklearn import kernel_ridge # For one exercise I will also be demonstrating Kernal Ridge Regression
from itertools import combinations
from sklearn.preprocessing import  MinMaxScaler # We will be using the imbuilt sclaing functions sklearn provides
from sklearn.preprocessing import OneHotEncoder # We will be using these to encode categorical features
from sklearn.impute import SimpleImputer # Performs basic imputations when doing preprocessing
import matplotlib.pyplot as plt # We will be using Matplotlib for our graphs
from sklearn.model_selection import train_test_split # A library that can automatically perform data splitting for us
from sklearn.metrics import mean_squared_error # Allows us to use the MSE function without calling in sklearn each time
import warnings

warnings.filterwarnings('ignore') # suppresses a convergence warning we may get when testing Lasso.
pd.options.mode.chained_assignment = None  # default='warn'
# you can use the read_csv command to store the .csv in a variable
housingData = pd.read_csv('housing_coursework_entire_dataset_23-24.csv')
# This file has all the data elements constructed already, so we can just put the entire thing into the 'data='
df = pd.DataFrame(data= housingData)
display(df)
print('Shape of the data (rows and columns):')
print(df.shape)
print()
print('List of the column names:')
print(df.columns)
print()
print('The data type of all the columns (all just floats here):')
print(df.dtypes)
print(df.describe())
df[df.columns[1:-1]].hist(alpha=0.5, figsize=(20, 10))
print('Original dataset length:')
print(len(df))
cleaner_df = df.drop_duplicates()
print('Dataset length after removing all rows with duplicates:')
print(len(cleaner_df)) # This action removed 0 items :. keep df
display(df.select_dtypes(include=np.number).describe())
display(df.select_dtypes(exclude=np.number).describe())

# Identify numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('No.')
numerical_features.remove('median_house_value')  # Exclude the target variable


# Calculate the number of rows and columns for subplots
num_features = len(numerical_features)
num_cols = 3  # You can adjust this number based on your preference
num_rows = (num_features + num_cols - 1) // num_cols  # Ensure enough rows

# Plot scatter graphs
plt.figure(figsize=(15, 5 * num_rows))
for i, feature in enumerate(numerical_features):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.scatterplot(x=df[feature], y=df['median_house_value'])
    plt.title(f'{feature} vs median_house_value')
    plt.xlabel(feature)
    plt.ylabel('median_house_value')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='ocean_proximity', y='median_house_value', data=df, marker='o', sort=False)
plt.title('Line Plot with Individual Data Points')
plt.xlabel('ocean_proximity')
plt.ylabel('median_house_value')
plt.show()

# Calculate mean of target variable for each category
category_means = df.groupby('ocean_proximity')['median_house_value'].median().reset_index()

# Plot line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='ocean_proximity', y='median_house_value', data=category_means, marker='o')
plt.title('Median of median_house_value for each ocean_proximity')
plt.xlabel('ocean_proximity')
plt.ylabel('Mean of median_house_value')
plt.show()


# Sort the data by category if necessary
df = cleaner_df.sort_values(by='ocean_proximity')

# Plot line plot with individual data points
plt.figure(figsize=(10, 6))
plt.plot(df['ocean_proximity'].astype(str), df['median_house_value'], marker='o', linestyle='-', markersize=6)
plt.title('Line Plot with Individual Data Points')
plt.xlabel('ocean_proximity')
plt.ylabel('median_house_value')
plt.show()

# Create a box plot
plt.figure(figsize=(10, 6))
df.boxplot(column='median_house_value', by='ocean_proximity')
plt.title('Box Plot of median_house_value by ocean_proximity')
plt.suptitle('')  # Suppress the automatic title to make the plot cleaner
plt.xlabel('ocean_proximity')
plt.ylabel('median_house_value')
plt.show()
