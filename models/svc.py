import os
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# load training & test from csv
x_train = pd.read_csv('data/train.csv')
x_test = pd.read_csv('data/test.csv')

# creating submission csv
submission = pd.DataFrame(
    columns=["sequence"], data=x_test["sequence"].astype(int))

# make the columns the time series
sensor_names = ["sensor_00", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "sensor_05", "sensor_06",
                "sensor_07", "sensor_08", "sensor_09", "sensor_10", "sensor_11", "sensor_12"]
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

# load training labels from csv
y_train = pd.read_csv('data/train_labels.csv')

# remove all but state data
y_train = y_train.loc[:, y_train.columns != 'sequence']

# remove multi index for model
x_train = x_train.drop(sensor_names, axis=1, level=0)
x_train = x_train.reset_index()

# # feature selection
# selector = SelectKBest(k=30, score_func=mutual_info_classif)
# mask = selector.get_support()
# features_chosen = x_train.columns[mask]
# x_train = selector.fit_transform(x_train.values, y_train.values.ravel())

model = BaggingClassifier(SVC(), n_jobs=-1)

selector = SequentialFeatureSelector(model, n_features_to_select=3)
selector.fit(x_train.values, y_train.values.ravel())
mask = selector.get_support()
features_chosen = x_train.columns[mask]
x_train = selector.fit_transform(x_train.values, y_train.values.ravel())

print("Finished SFS")
print("Features chosen:", features_chosen)
# adjust test for features chosen
x_test = x_test[features_chosen]

# # z scaling
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)

# model
model = BaggingClassifier(
    SVC(), n_jobs=-1).fit(x_train, y_train.values.ravel())
print("Finished training model")
print("Accuracy: ", np.mean(cross_val_score(
    model, x_train, y_train.values.ravel(), cv=10)))

# # predict y_test
y_pred = model.predict(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/svc', exist_ok=True)
submission.to_csv('submissions/svc/out.csv', index=False)
