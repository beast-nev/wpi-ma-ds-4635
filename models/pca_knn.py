from collections import Counter
from itertools import combinations
import os
from statistics import mode
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, normalize, minmax_scale

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
y_train = y_train.sample(frac=1.0, random_state=42)
x_test = x_test.sample(frac=1.0, random_state=42)

# z transformation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# feature selection & model creation
model = KNeighborsClassifier(n_neighbors=5)

# impute nans
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
x_train = imp.fit_transform(x_train)
x_test = imp.fit_transform(x_test)

# pca
pca = PCA()
pca.fit(x_train, y_train.values.ravel())

# transform x_train for training
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# fitting for prediction
model.fit(x_train, y_train.values.ravel())

# predict training values for training scoring
y_pred_train = model.predict(x_train)

# compute roc_auc and classification report
print("Roc score: ", roc_auc_score(y_train, y_pred_train))
print("Classification report: ", classification_report(
    y_true=y_train, y_pred=y_pred_train))

# test prediction
y_pred = model.predict_proba(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred[:, 1]

# write to csv for kaggle submission
os.makedirs('submissions/pca_knn', exist_ok=True)
submission.to_csv('submissions/pca_knn/out.csv', index=False)
