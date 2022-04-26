from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import pandas as pd
from scipy import rand
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample


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
    x_train[i+"_count"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).count()
    x_train[i+"_cummax"] = x_train_load[i].groupby(
        np.arange(len(x_train_load[i])) // 60).cummax()
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
    x_test[i+"_count"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).count()
    x_test[i+"_cummax"] = x_test_load[i].groupby(
        np.arange(len(x_test_load[i])) // 60).cummax()
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

print("Finished creating features")

# # sample
# x_train = x_train.sample(frac=1.0, random_state=42)

# feature selection & model creation
mlp = MLPClassifier(max_iter=100, solver="adam")
mlp_space = {
    'hidden_layer_sizes': [(10, 30, 10), (30, 75, 20), (50, 100, 35), (20,), (50,), ],
    'activation': ['relu', 'logistic'],
    'alpha': [0.0001, 0.05, 0.1, 0.25],
    'learning_rate': ['constant', 'adaptive'],
}
grid_search = GridSearchCV(mlp, mlp_space, n_jobs=-1, cv=5)
# X is train samples and y is the corresponding labels
grid_search.fit(x_train, y_train)

# # pca
# pca = PCA(0.95, random_state=42)
# pca.fit(x_train, y_train.values.ravel())
# # print("Explained Variance ratio:", pca.explained_variance_ratio_)

# # transform x_train for training
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)

# print("Finished feature selection")

# print("Accuracy: ", np.mean(cross_val_score(
#     model, x_train, y_train.values.ravel(), cv=5, n_jobs=-1)))

# # fitting for prediction
# model.fit(x_train, y_train.values.ravel())

y_pred_train = grid_search.predict(x_train)

print("Average precision score: ", average_precision_score(y_train, y_pred_train))
print("Roc score: ", roc_auc_score(y_train, y_pred_train))

# predict y_test
y_pred = grid_search.predict(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/boosting', exist_ok=True)
submission.to_csv('submissions/boosting/out.csv', index=False)

# No PCA
# Accuracy:  0.798559781869213
# Average precision score:  0.7820865392398746
# Roc score:  0.833622885177769

# Accuracy:  0.760897088962134
# Average precision score:  0.7492426610591539
# Roc score:  0.8207008229210955

# With PCA
# Accuracy:  0.6461024626542899
# Average precision score:  0.6057154692572313
# Roc score:  0.6549551915893461

# Accuracy:  0.5037356128154803
# Average precision score:  0.5220779326228184
# Roc score:  0.5400416219024612
