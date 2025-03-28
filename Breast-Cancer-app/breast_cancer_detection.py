# -*- coding: utf-8 -*-
"""Breast cancer detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HCGhLyLbDwRqVY_697ToXZePclm2WWfM

**Breast Cancer Detection Using Machine Learning**
"""

#python module downloads
!pip  install numpy
!pip install pandas
!pip install matplotlib
!pip install seaborn

#importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dataframe as df
df = pd.read_csv('/content/data.csv')

df.head()

# from google.colab import files
# upload = files.upload()

# df_1 = pd.read_csv('/content/data.csv')

# df_1.head()

# #installing the kaggle library
# ! pip install kaggle
# #make a directory named ".kaggle"
# ! mkdir ~/ .kaggle
# #copy the "kaggle.json" into this new directory
# ! cp kaggle.json ~/ .kaggle/
# #allocating the required permissions for this file
# ! chmod 600 ~/ .kaggle/kaggle.json

# #downloading the datasets
# ! kaggle datasets download uciml/breast-cancer-wisconsin-data

# #unzip the file
# ! unzip breast-cancer-wisconsin-data.zip

# df_1.head()

# EDA : Exploratory data analysis
#checking total number of rows and columns
df.shape

#checking the data and their corresponding data types
#the properties of the data - summary of the statistics
df.info()

#2nd way for checking null values
df.isnull().sum()

#drop the column with all missing values
# axis = 1: columns and axis = 2: rows
df = df.dropna(axis=1)

df.shape

#checking datatypes
df.dtypes

# display the count for objects, here objects are benign(B) and malignant(M)
df['diagnosis'].value_counts()

#visual represention of diagnosis
sns.countplot(df['diagnosis'], label = 'count')

#Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()

#transforming categorical data to numerical
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)

#printing the numerical values, M = 1, B = 0
df.iloc[:,1].values

sns.pairplot(df.iloc[:,1:7], hue = 'diagnosis')

#co-relation between columns
df.iloc[:,1:11].corr()

#heatmap
plt.figure(figsize = (10,10))
sns.heatmap(df.iloc[:,1:11].corr(), cmap ="rocket", annot = True, fmt = ".0%") # other color palletes are YlGnBu, rocket_r, tab10, etc

#feature scalling
#split our dataset into independent and dependent datasets
#independent --> X
#dependent --> Y
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#80:20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print(Y_train[:5])  # First 5 elements
print(np.unique(Y_train))  # Unique values in the array
print(Y_train.dtype)  # Data type of the array
Y_train = Y_train.astype(int)  # Convert to integers

X_train

# Importing modules
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def models(X_train, Y_train):
    # Logistic regression classifier
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    # Decision tree classifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    # Random forest classifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    # Print the accuracy of each model on the training dataset
    print("The accuracy of Logistic Regression: ", log.score(X_train, Y_train))
    print("The accuracy of Decision Tree: ", tree.score(X_train, Y_train))
    print("The accuracy of Random Forest: ", forest.score(X_train, Y_train))

    return log, tree, forest

model = models(X_train, Y_train)

from sklearn.metrics import confusion_matrix

# Ensure predictions are binary
predictions = (model[0].predict(X_test) > 0.5).astype(int)

# Ensure Y_test and predictions are compatible
Y_test = Y_test.astype(int)
predictions = predictions.astype(int)

# Check for shape consistency
Y_test = Y_test.ravel()
predictions = predictions.ravel()

# Compute confusion matrix
cm = confusion_matrix(Y_test, predictions)

# Extract confusion matrix components
tp = cm[0][0]
tn = cm[1][1]
fn = cm[1][0]
fp = cm[0][1]

# Print confusion matrix and accuracy
print(cm)
print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('Model: ',i)
  print(classification_report(Y_test, model[i].predict(X_test)))
  print(accuracy_score(Y_test, model[i].predict(X_test)))
  print()

#prediction
pred = model[2].predict(X_test)
print('Our model prediction: ')
print(pred)
print()
print('Actual prediction: ')
print(Y_test)