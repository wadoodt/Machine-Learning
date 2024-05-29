# Import NumPy, which can deal with multi-dimensional arrays such as matrix intuitively.
import numpy as np # A useful package for dealing with mathematical processes, we will be using it this week for vectors and matrices
import pandas as pd # A common package for viewing tabular data
import matplotlib.pyplot as plt # We will be using Matplotlib for our graphs
import seaborn as sns; sns.set_style  # for plot styling
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score # required for evaluating classification models
from sklearn.model_selection import train_test_split # A library that can automatically perform data splitting for us
import sklearn.linear_model, sklearn.datasets # sklearn is an important package for much of the ML we will be doing, this time we are using the Linear Regression Model and the datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

titanic_data = pd.read_csv('Titanic_coursework_entire_dataset_23-24.csv')
df = pd.DataFrame(data= titanic_data)
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

# Plot the distribution of age based on survival
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Fare', hue='Survival', multiple='dodge', shrink=0.8, palette={0: 'red', 1: 'blue'})
plt.xlabel('Fare')
plt.ylabel('Number of Appearances')
plt.title('Distribution of Fare based on Survival')
plt.legend(title='Survival', labels=['Not Survived', 'Survived'])
plt.show()

# Plot the distribution of age based on survival
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survival', multiple='dodge', shrink=0.8, palette={0: 'red', 1: 'blue'})
plt.xlabel('Age')
plt.ylabel('Number of Appearances')
plt.title('Distribution of Age based on Survival')
plt.legend(title='Survival', labels=['Not Survived', 'Survived'])
plt.show()

features = ['Pclass', 'Sex','Age','SibSp','Parch','Fare','Embarked','Survival']
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

categFeatures = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
numFeatures = ['Age', 'Fare']

for col in df[categFeatures].columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, hue='Survival', data=df, palette={0: 'red', 1: 'blue'})
    plt.title(f'Relationship between {col} and Survival')
    plt.show()

