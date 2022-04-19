import os
from turtle import st
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


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

selector = SelectKBest(f_classif, k=3)
x_train = selector.fit_transform(x_train, y_train.values.ravel())

# With k=6 best
# Accuracies:  0.6365735885830673
# Test train split score:  0.6236683352586317

# Without k best
# Accuracies:  0.6643562389251304
# Test train split score:  0.6563984084199718

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train.values.ravel(), test_size=0.3, random_state=42)

# fit model
model = SVC(shrinking=True, probability=False, kernel='rbf', gamma='scale', degree=1.0, decision_function_shape='ovr', cache_size=580, break_ties=False, C=1.0).fit(
    X_train, Y_train)

# 'shrinking': True, 'probability': False, 'kernel': 'rbf', 'gamma': 'scale', 'degree': 1.0,
# 'decision_function_shape': 'ovr', 'cache_size': 579.5918367346939, 'break_ties': False, 'C': 1.0


print("Accuracies: ", np.mean(cross_val_score(
    model, X_train, Y_train, cv=5)))

# # predict y_test
Y_pred = model.predict(X_test)
print("Test train split score: ", accuracy_score(Y_test, Y_pred))

# make state in submission csv our prediction
# submission["state"] = y_pred

# # write to csv for kaggle submission
# os.makedirs('submissions/svc', exist_ok=True)
# submission.to_csv('submissions/svc/out.csv', index=False)
