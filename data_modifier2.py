import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\wojciech.maludzinski\\Repos\\download\\btcusd-m1-bid-2019-01-01-2022-01-01.csv",
                 header=[0])
last = df.iloc[:, -1]
# last['close'] = last['close'].astype(float)
print(last)
lastlist = last.values.tolist()
print(lastlist)


def split_list(_list, group_size, overlap_size):
    return [_list[i:i + group_size] for i in range(0, len(_list), group_size - overlap_size)]


nicelist = split_list(lastlist, 1000, 995)
print(nicelist[1:100])



# def split_dataframe(df, chunk_size=1000):
#     chunks = list()
#     num_chunks = len(df) // chunk_size + 1
#     for i in range(num_chunks):
#         chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
#     return chunks


# big_array = []
# chunks = split_dataframe(last[1:])
#
# for c in chunks:
#     array = []
#     for e in c:
#         array.append(float(e))
#     big_array.append(np.asarray(array))
#
#
# def split_dataframe_overlap(df, chunk_size=1000, overlap=995):
#     chunks = list()
#     num_chunks = (len(df) // chunk_size + 1) * chunk_size//(chunk_size - overlap)
#     for i in range(num_chunks):
#         chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
#
#     return chunks
#
#
# big_array = []
# chunks = split_dataframe(last[1:])
#
# for c in chunks:
#     array = []
#     for e in c:
#         array.append(float(e))
#     big_array.append(np.asarray(array))
#
dataframe = pd.DataFrame(nicelist, columns=None)
dataframe.to_csv('data\\array_btc2.csv', index=False, header=False)
