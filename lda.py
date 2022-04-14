import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

x_train = pd.read_csv('data/train.csv')
# print("Number of rows in train:", x_train.shape[0])

y_train = pd.read_csv('data/train_labels.csv')
# print("Number of rows in test:", y_train.shape[0])

# print(x_train.shape[0] / y_train.shape[0])  # evenly divisible data points

# data summary attempt
x_train[["step"]] = pd.to_timedelta(x_train[["step"]], unit='S')
x_train.resample('S', on='step').mean()
print(x_train[["step"]])
