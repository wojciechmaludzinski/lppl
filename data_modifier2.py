import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\wojciech.maludzinski\\Repos\\download\\usdpln-h1-bid-2008-01-01-2009-01-01.csv",
                 header=[0])
last = df.iloc[:, -1]
lastlist = last.values.tolist()



def split_list(_list, group_size, overlap_size):
    return [_list[i:i + group_size] for i in range(0, len(_list), group_size - overlap_size)]


nicelist = split_list(lastlist, 1000, 995)
# print(nicelist[1:100])

dataframe = pd.DataFrame(nicelist, columns=None)
dataframe.to_csv('data\\array_usdpln.csv', index=False, header=False)
