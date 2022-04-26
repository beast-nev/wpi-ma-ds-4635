import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.impute import SimpleImputer

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

# features names for pca
feature_names = x_train.columns

# z transformation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# feature selection & model creation
model = HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=31,
                                       max_iter=1500, min_samples_leaf=250,
                                       l2_regularization=1,
                                       random_state=42, verbose=0, scoring="roc_auc")

# impute nans
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
x_train = imp.fit_transform(x_train)
x_test = imp.fit_transform(x_test)

# pca
pca = PCA()
pca.fit(x_train, y_train.values.ravel())
# print("Explained Variance ratio:", pca.explained_variance_ratio_)

# number of components
n_pcs = pca.components_.shape[0]

# transform x_train for training
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

# get the names
most_important_names = [
    feature_names[most_important[i]] for i in range(n_pcs)]

# LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.DataFrame(dic.items())

# print("Accuracy: ", np.mean(cross_val_score(
#     model, x_train, y_train.values.ravel(), cv=5, n_jobs=-1)))

# fitting for prediction
model.fit(x_train, y_train.values.ravel())

# predict training values for training scoring
y_pred_train = model.predict(x_train)

# compute roc_auc and classification report
print("Roc score: ", roc_auc_score(y_train, y_pred_train))
print("Classification report: ", classification_report(
    y_true=y_train, y_pred=y_pred_train))

# test predcition
y_p = model.predict(x_test)

# predict y_test probability
y_pred = pd.DataFrame(data=model.predict_proba(x_test),
                      columns=["state0", "state1"])
y_pred["pred"] = np.max(y_pred.values, axis=1)

# make state in submission csv our prediction
submission["state"] = y_pred["pred"]

# write to csv for kaggle submission
os.makedirs('submissions/boosting', exist_ok=True)
submission.to_csv('submissions/boosting/out.csv', index=False)
