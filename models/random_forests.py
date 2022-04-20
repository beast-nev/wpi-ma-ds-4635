import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time

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

# n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=5)]
# max_features = ['sqrt']
# max_depth = [int(x) for x in np.linspace(3, 49, num=2)]
# min_samples_split = [3, 4, 6, 7, 8, 10, 12, 16, 22]
# min_samples_leaf = [1, 2, 3, 4, 5, 6, 7]
# bootstrap = [True]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# rf = RandomForestClassifier()
# start_search = time.time()
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
#                                n_iter=100, cv=5, verbose=0, random_state=42, n_jobs=-1)
# rf_random.fit(x_train, y_train.values.ravel())
# end_search = time.time()
# print("Took: ", end_search-start_search, " seconds to find best params")
# print(rf_random.best_params_)

# 'n_estimators': 466, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': True
# 'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 49, 'bootstrap': True
# model
model = RandomForestClassifier(n_estimators=500, min_samples_split=2,
                               min_samples_leaf=1, max_features='sqrt', max_depth=60, bootstrap=True)

print("Accuracies: ", np.mean(cross_val_score(
    model, x_train, y_train.values.ravel(), cv=5)))
model.fit(x_train, y_train.values.ravel())

# predict y_test
y_pred = model.predict(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/random_forests', exist_ok=True)
submission.to_csv('submissions/random_forests/out.csv', index=False)
