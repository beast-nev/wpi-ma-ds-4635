import os
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier

# load training & test from csv
x_train = pd.read_csv('data/train.csv')
# x_test = pd.read_csv('data/test.csv')

# aggregate each 60 seconds by averaging sensor data
x_train = x_train.groupby(np.arange(len(x_train)) // 60).mean()
# x_test = x_test.groupby(np.arange(len(x_test)) // 60).mean()

# creating submission csv
# submission = pd.DataFrame(
#     columns=["sequence"], data=x_test["sequence"].astype(int))

# remove all but sensor data
x_train = x_train.loc[:, ~x_train.columns.isin(
    ['sequence', 'subject', 'step'])]
# x_test = x_test.loc[:, ~x_test.columns.isin(
#     ['sequence', 'subject', 'step'])]

# load training labels from csv
y_train = pd.read_csv('data/train_labels.csv')

# remove all but state data
y_train = y_train.loc[:, y_train.columns != 'sequence']

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train.values.ravel(), test_size=0.3, random_state=42)

nca = NeighborhoodComponentsAnalysis(n_components=3, random_state=42)
x_train = nca.fit_transform(X_train, Y_train)
model = KNeighborsClassifier(n_neighbors=3)

# print("Accuracies: ", np.mean(cross_val_score(
#     model, x_train, y_train.values.ravel(), cv=5)))
model.fit(X_train, Y_train)
print("Accuracy: ", model.score(X_test, Y_test))
