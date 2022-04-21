import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SequentialFeatureSelector
from sklearn.preprocessing import MultiLabelBinarizer

# load training & test from csv
x_train = pd.read_csv('data/train.csv')
x_test = pd.read_csv('data/test.csv')

# make the columns the time series
sensor_names = ["sensor_00", "sensor_01", "sensor_02", "sensor_03", "sensor_04", "sensor_05", "sensor_06",
                "sensor_07", "sensor_08", "sensor_09", "sensor_10", "sensor_11", "sensor_12"]

# creating submission csv
submission = pd.DataFrame(
    columns=["sequence"], data=x_test[["sequence"]].groupby(np.arange(len(x_test[["sequence"]])) // 60).mean())

x_train = x_train.pivot(
    index=["sequence", "subject"], columns="step", values=sensor_names)
x_test = x_test.pivot(
    index=["sequence", "subject"], columns="step", values=sensor_names)


# create new features for the mean,std,var,max,min,and sum of each sensor values
for i in sensor_names:
    x_train[i+"_mean"] = x_train[i].mean(axis=1)
    x_train[i+"_std"] = x_train[i].std(axis=1)
    x_train[i+"_var"] = np.power(x_train[i].std(axis=1), 2)
    x_train[i+"_max"] = x_train[i].max(axis=1)
    x_train[i+"_min"] = x_train[i].min(axis=1)
    x_train[i+"_sum"] = x_train[i].sum(axis=1)

    x_test[i+"_mean"] = x_test[i].mean(axis=1)
    x_test[i+"_std"] = x_test[i].std(axis=1)
    x_test[i+"_var"] = np.power(x_test[i].std(axis=1), 2)
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

# feature selection

# remove subject from features
x_train = x_train.drop("subject", axis=1, level=0)
x_test = x_test.drop("subject", axis=1, level=0)

# select k best features based on mutual_classfi
selector = SelectKBest(k=64, score_func=f_classif)
selector.fit(x_train, y_train.values.ravel())

# get which features we want for test
mask = selector.get_support()
features_chosen_multi_index = x_train.columns[mask]
features_chosen = [feature_tuple[0]
                   for feature_tuple in features_chosen_multi_index]

# transform x_train for training
x_train = selector.fit_transform(x_train.values, y_train.values.ravel())

# adjust test for features chosen
x_test = x_test.drop(sensor_names, axis=1, level=0)
x_test = x_test.reset_index()
x_test = x_test[features_chosen]
x_test = np.array(x_test)

# z scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

print("Finished feature selection")

# loss = ["deviance", "exponential"]
# n_estimators = np.linspace(50, 300, 5).astype(int)
# criterion = ["friedman_mse"]
# min_samples_split = np.linspace(2, 6, 6).astype(int)
# min_samples_leaf = np.linspace(1, 5, 5).astype(int)
# max_depth = np.linspace(3, 9, 3).astype(int)
# max_features = ["sqrt", "log2"]

# random_grid = {
#     'loss': loss,
#     'n_estimators': n_estimators,
#     'criterion': criterion,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'max_depth': max_depth,
#     'max_features': max_features
# }

# model
# (verbose=0, random_state=42, n_estimators=300, min_samples_split=3,min_samples_leaf=4, max_features="sqrt", max_depth=9, loss="exponential", criterion="friedman_mse")
# model = GradientBoostingClassifier(verbose=1, random_state=42, n_estimators=300, min_samples_split=3,
#                                    min_samples_leaf=4, max_features="sqrt", max_depth=9, loss="exponential", criterion="friedman_mse")
model = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=25,
                                       max_iter=1000, min_samples_leaf=500,
                                       l2_regularization=1,
                                       max_bins=255,
                                       random_state=4, verbose=0)

# boosting_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
#                                      n_iter=100, cv=5, verbose=1, random_state=42, n_jobs=4)

# boosting_random.fit(x_train, y_train.values.ravel())
# print(boosting_random.best_params_)
# print(boosting_random.best_score_)

# model scoring
print("Accuracy: ", np.mean(cross_val_score(
    model, x_train, y_train.values.ravel(), cv=10, verbose=1)))

# fitting for prediction
model.fit(x_train, y_train.values.ravel())

# predict y_test
y_pred = model.predict(x_test)

# make state in submission csv our prediction
submission["state"] = y_pred

# write to csv for kaggle submission
os.makedirs('submissions/boosting', exist_ok=True)
submission.to_csv('submissions/boosting/out.csv', index=False)
