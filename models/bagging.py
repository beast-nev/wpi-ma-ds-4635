from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
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

# z scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# fit model
model = BaggingClassifier(KNeighborsClassifier(
    n_neighbors=3), max_samples=0.3, max_features=0.75)

# model scoring
print("Accuracy: ", np.mean(cross_val_score(
    model, x_train, y_train.values.ravel(), cv=10)))

# fitting for prediction
model.fit(x_train, y_train.values.ravel())

# predict y_test
y_pred = model.predict(x_test.values)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/bagging', exist_ok=True)
submission.to_csv('submissions/bagging/out.csv', index=False)
