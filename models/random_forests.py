import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# load training & test from csv
x_train = pd.read_csv('data/train.csv')
x_test = pd.read_csv('data/test.csv')

# aggregate each 60 seconds by averaging sensor data
x_train = x_train.groupby(np.arange(len(x_train)) // 60).mean()
x_test = x_test.groupby(np.arange(len(x_test)) // 60).mean()

# creating submission csv
submission = pd.DataFrame(
    columns=["sequence"], data=x_test["sequence"].astype(int))

# remove all but sensor data
x_train = x_train.loc[:, ~x_train.columns.isin(
    ['sequence', 'subject', 'step'])]
x_test = x_test.loc[:, ~x_test.columns.isin(
    ['sequence', 'subject', 'step'])]

# load training labels from csv
y_train = pd.read_csv('data/train_labels.csv')

# remove all but state data
y_train = y_train.loc[:, y_train.columns != 'sequence']

# feature selection
selector = SelectKBest(mutual_info_classif, k=6)
x_train = selector.fit_transform(x_train, y_train.values.ravel())

# if_classif
# Accuracies:  0.6850963844665071
# Test train split score:  0.6733410345270183

# mustal_class_if
# Accuracies:  0.6935688658274356
# Test train split score:  0.6920806058272365

# predictors choosen
# print(selector.get_feature_names_out())
# 'sensor_02' 'sensor_04' 'sensor_06' 'sensor_08' 'sensor_10' 'sensor_12'

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train.values.ravel(), test_size=0.3, random_state=42)

# fit model
model = RandomForestClassifier(n_estimators=250, min_samples_split=5,
                               min_samples_leaf=2, max_features='sqrt', max_depth=20, bootstrap=True)
sfs = SequentialFeatureSelector(model, n_features_to_select=3)
sfs.fit(X_train, Y_train)
print(sfs.get_feature_names_out())
print(sfs.get_support())
print("Accuracies: ", np.mean(cross_val_score(
    model, X_train, Y_train, cv=5)))
model.fit(X_train, Y_train)

# predict y_test
y_pred = model.predict(X_test)

print("Test train split score: ", accuracy_score(Y_test, y_pred))
# make state in submission csv our prediction
# submission["state"] = y_pred

# # write to csv for kaggle submission
# os.makedirs('submissions/random_forests', exist_ok=True)
# submission.to_csv('submissions/random_forests/out.csv', index=False)
