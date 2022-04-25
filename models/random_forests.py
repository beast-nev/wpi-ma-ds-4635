import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score

# load training & test from csv
x_train_load = pd.read_csv('data/train.csv')
x_test_load = pd.read_csv('data/test.csv')

# load training labels from csv
y_train_load = pd.read_csv('data/train_labels.csv')

# remove all but state data
y_train = y_train_load.loc[:, y_train_load.columns != 'sequence']

# sensor names for easy indexing
sensor_names = ["sensor_00", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "sensor_05", "sensor_06",
                "sensor_07", "sensor_08", "sensor_09", "sensor_10", "sensor_11", "sensor_12"]

# creating submission csv
submission = pd.DataFrame(
    columns=["sequence"], data=x_test_load[["sequence"]].groupby(np.arange(len(x_test_load[["sequence"]])) // 60).mean())

# create new features for the mean,std,max,min,and sum of each sensor values
x_train = pd.DataFrame()
x_test = pd.DataFrame()

for i in sensor_names:
    x_train[i+"_mean"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).mean()
    x_train[i+"_std"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).std()
    x_train[i+"_max"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).max()
    x_train[i+"_min"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).min()
    x_train[i+"_sum"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).sum()
    x_train[i+"_median"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).median()
    x_train[i+"_q1"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).quantile(0.25)
    x_train[i+"_q3"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).quantile(0.75)

    x_test[i+"_mean"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).mean()
    x_test[i+"_std"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).std()
    x_test[i+"_max"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).max()
    x_test[i+"_min"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).min()
    x_test[i+"_sum"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).sum()
    x_test[i+"_median"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).median()
    x_test[i+"_q1"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).quantile(0.25)
    x_test[i+"_q3"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).quantile(0.75)

print(x_train.head(3))
print(x_train.shape)
print(x_test.head(3))
print(x_test.shape)

# feature selection & model creation
model = RandomForestClassifier(
    n_estimators=500, verbose=0, n_jobs=-1, max_depth=25)

# forward subset selection
start_time = time()
selector = SequentialFeatureSelector(
    model, direction="forward", n_features_to_select=3).fit(x_train, y_train.values.ravel())
end_time = time()

# runtime of subset selection
print("Total selection time: ", end_time-start_time)

# get which features we want for test
mask = selector.get_support()
features_chosen_mask = x_train.columns[mask]
features_chosen = [feature
                   for feature in features_chosen_mask]
print("Features chosen: ", features_chosen)

# transform x_train for training
x_train = selector.transform(x_train)

x_test = x_test[features_chosen]

# z scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print("Finished feature selection")

print("Accuracy: ", np.mean(cross_val_score(
    model, x_train, y_train.values.ravel(), cv=5, n_jobs=-1)))

model.fit(x_train, y_train.values.ravel())

y_pred_train = model.predict(x_train)

print("Average precision score: ", average_precision_score(y_train, y_pred_train))
print("Roc score: ", roc_auc_score(y_train, y_pred_train))

# predict y_test
y_pred = model.predict(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/random_forests', exist_ok=True)
submission.to_csv('submissions/random_forests/out.csv', index=False)
