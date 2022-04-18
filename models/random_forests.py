from turtle import pd
import numpy as np
import pandas as pd

x_train = pd.read_csv('data/train.csv')
x_train = x_train.groupby(np.arange(len(x_train)) // 60).mean()
print(x_train)
