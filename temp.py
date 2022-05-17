import pandas as pd
import csv
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from tensorflow.python.client import device_lib
#
#
# if __name__ == '__main__':
#     print(device_lib.list_local_devices())

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))
tf.config.experimental.set_memory_growth(gpu[0], True)