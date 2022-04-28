import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold

from models.boosting import X_train


# load training & test from csv
x_train_load = pd.read_csv('data/train.csv')
x_test_load = pd.read_csv('data/test.csv')

# load training labels from csv
y_train_load = pd.read_csv('data/train_labels.csv')

# remove all but state data
y_train = y_train_load.loc[:, y_train_load.columns != 'sequence']
y_train_index = y_train
y_train = y_train.values.ravel()

# sensor names for easy indexing
sensor_names = ["sensor_00", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "sensor_05", "sensor_06",
                "sensor_07", "sensor_08", "sensor_09", "sensor_10", "sensor_11", "sensor_12"]

# creating submission csv
submission = pd.DataFrame(
    columns=["sequence"], data=x_test_load[["sequence"]].groupby(np.arange(len(x_test_load[["sequence"]])) // 60).mean())

# create new features for the mean,std,max,min,and sum of each sensor values
x_train = pd.DataFrame()
x_test = pd.DataFrame()

# create features for mean, lag, std, min, max, median, and iqr
for i in sensor_names:
    x_train[i+"_mean"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).mean()
    x_train[i+"_lag"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).shift(1)
    x_train[i+"_std"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).std()
    x_train[i+"_max"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).max()
    x_train[i+"_min"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).min()
    x_train[i+"_median"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).median()
    x_train[i+"_iqr"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).quantile(0.75) - x_train_load[i].groupby(np.arange(len(x_train_load[i])) // 60).quantile(0.25)

    x_test[i+"_mean"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).mean()
    x_test[i+"_lag"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).shift(1)
    x_test[i+"_std"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).std()
    x_test[i+"_max"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).max()
    x_test[i+"_min"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).min()
    x_test[i+"_median"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).median()
    x_test[i+"_irq"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).quantile(0.75) - x_test_load[i].groupby(np.arange(len(x_test_load[i])) // 60).quantile(0.25)

# take samples of our data
x_train = x_train.sample(frac=1.0, random_state=42)
# y_train = y_train.sample(frac=1.0, random_state=42)
x_test = x_test.sample(frac=1.0, random_state=42)

# features names for pca
feature_names = x_train.columns

# z transformation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# feature selection & model creation
model = RandomForestClassifier(
    n_estimators=500, verbose=0, n_jobs=-1, max_depth=15)

# impute nans
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
x_train = imp.fit_transform(x_train)
x_test = imp.fit_transform(x_test)

# pca
pca = PCA()
pca.fit(x_train, y_train)
# print("Explained Variance ratio:", pca.explained_variance_ratio_)

# transform x_train for training
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# group_kfold = GroupKFold()

# bestScore = 0
# bestPred = []

# for train_index, test_index in group_kfold.split(x_train, y_train, y_train_index.index):
#     X_train, X_test = x_train[train_index], x_train[test_index]
#     Y_train, Y_test = y_train[train_index], y_train[test_index]

#     # fit model
#     model.fit(X_train, Y_train)

#     # predict val auc_roc score
#     y_pred_val = model.predict_proba(X_test)

#     # compute roc_auc and classification report
#     score = roc_auc_score(Y_test, y_pred_val[:, 1])
#     print("Roc score: ", roc_auc_score(Y_test, y_pred_val[:, 1]))
#     if score > bestScore:
#         bestScore = score
#         print("New best score: ", bestScore)

#         # test prediction
#         bestPred = model.predict_proba(x_test)[:, 1]


# print("Classification report: ", classification_report(
#     y_true=Y_test, y_pred=y_pred_val))

X_train, X_test, Y_train, Y_test = train_test_split(
    x_train, y_train, test_size=0.33, random_state=42)

model.fit(X_train, Y_train)

# predict y_test probability
y_pred = model.predict_proba(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred[:, 1]

# write to csv for kaggle submission
os.makedirs('submissions/random_forests', exist_ok=True)
submission.to_csv('submissions/random_forests/out.csv', index=False)
