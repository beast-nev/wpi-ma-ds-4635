import os
from sklearn.svm import SVC
import numpy as np
import pandas as pd

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

# fit model
model = SVC().fit(
    x_train, y_train.values.ravel())

# predict y_test
y_pred = model.predict(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/svc', exist_ok=True)
submission.to_csv('submissions/svc/out.csv', index=False)