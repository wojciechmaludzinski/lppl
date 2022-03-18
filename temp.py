import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data\\arrays.csv", header=None)
df = df.sample(frac=1).reset_index(drop=True)


y = df[df.columns[0]]
X = df.iloc[: , 1:]

X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.2)