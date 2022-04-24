import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time

# load training & test from csv
x_train = pd.read_csv('data/train.csv')
x_test = pd.read_csv('data/test.csv')

# sensor names for easy indexing
sensor_names = ["sensor_00", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "sensor_05", "sensor_06",
                "sensor_07", "sensor_08", "sensor_09", "sensor_10", "sensor_11", "sensor_12"]

# creating submission csv
submission = pd.DataFrame(
    columns=["sequence"], data=x_test[["sequence"]].groupby(np.arange(len(x_test[["sequence"]])) // 60).mean())

# change index of train and test
x_train = x_train.pivot(
    index=["sequence", "subject"], columns="step", values=sensor_names)
x_test = x_test.pivot(
    index=["sequence", "subject"], columns="step", values=sensor_names)

# create new features for the mean,std,max,min,and sum of each sensor values
for i in sensor_names:
    x_train[i+"_mean"] = x_train[i].mean(axis=1)
    x_train[i+"_std"] = x_train[i].std(axis=1)
    x_train[i+"_max"] = x_train[i].max(axis=1)
    x_train[i+"_min"] = x_train[i].min(axis=1)
    x_train[i+"_sum"] = x_train[i].sum(axis=1)

    x_test[i+"_mean"] = x_test[i].mean(axis=1)
    x_test[i+"_std"] = x_test[i].std(axis=1)
    x_test[i+"_max"] = x_test[i].max(axis=1)
    x_test[i+"_min"] = x_test[i].min(axis=1)
    x_test[i+"_sum"] = x_test[i].sum(axis=1)

# load training labels from csv
y_train = pd.read_csv('data/train_labels.csv')

# remove all but state data
y_train = y_train.loc[:, y_train.columns != 'sequence']

# remove multi index for model
x_train = x_train.drop(sensor_names, axis=1, level=0)
x_train = x_train.reset_index()

# remove multi index for model
x_test = x_test.drop(sensor_names, axis=1, level=0)
x_test = x_test.reset_index()

# feature selection & model creation
model = RandomForestClassifier(
    n_estimators=300, verbose=3, n_jobs=-1, max_depth=10)

# remove subject from features
x_train = x_train.drop("subject", axis=1, level=0)
x_test = x_test.drop("subject", axis=1, level=0)

# forward subset selection
start_time = time()
selector = SequentialFeatureSelector(
    model, direction="forward").fit(x_train, y_train.values.ravel())
end_time = time()

# runtime of subset selection
print("Total selection time: ", end_time-start_time)

# remove subject from features
x_train = x_train.drop("subject", axis=1, level=0)
x_test = x_test.drop("subject", axis=1, level=0)

# get which features we want for test
mask = selector.get_support()
features_chosen_multi_index = x_train.columns[mask]
features_chosen = [feature_tuple[0]
                   for feature_tuple in features_chosen_multi_index]
print("Features chosen: ", features_chosen)

# transform x_train for training
x_train = selector.transform(x_train)

# adjust test for features chosen
x_test = x_test.drop(sensor_names, axis=1, level=0)
x_test = x_test.reset_index()
x_test = x_test[features_chosen]
x_test = np.array(x_test)

# z scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

print("Finished feature selection")

# model
print("Accuracy: ", np.mean(cross_val_score(
    model, x_train, y_train.values.ravel(), cv=4)))

model.fit(x_train, y_train.values.ravel())

# predict y_test
y_pred = model.predict(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/random_forests', exist_ok=True)
submission.to_csv('submissions/random_forests/out.csv', index=False)
