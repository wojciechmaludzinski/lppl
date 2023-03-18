import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D

PATH = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 0=GPU, -1=CPU

if tf.config.list_physical_devices():
    print("GPU is available")
else:
    print("GPU is not available")

# Import libraries


# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)
print(data_dir_list)

img_rows = 128
img_cols = 128
num_channel = 1
num_epoch = 20

# Define number of classes
num_classes = 2

labels_name = {'noise': 0, 'lppl': 1}

img_data_list = []
labels_list = []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loading the images of dataset-' + '{}\n'.format(dataset))
    label = labels_name[dataset]
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (128, 128))
        img_data_list.append(input_img_resize)
        labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255

print(img_data.shape)
labels = np.array(labels_list)

img_data = np.expand_dims(img_data, axis=3)
print(img_data.shape)

print(np.unique(labels, return_counts=True))

# Convert class labels to on-hot encoding
# Y = tf.keras.utils.to_categorical(labels, num_classes)
Y = labels

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print("X_train shape = {}".format(X_train.shape))
print("X_test shape = {}".format(X_test.shape))

# Initializing the input shape
input_shape = img_data[0].shape
print('input shape:', img_data[0].shape)

# Design CNN sequential model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #svg albo cos innego, knn, popatrzeć
model.compile(tf.keras.optimizers.RMSprop(lr=0.0005, decay=1e-6), loss="sparse_categorical_crossentropy", )
model.summary()

run_hist_1 = model.fit(X_train, y_train, epochs=15,
                       validation_data=(X_test, y_test),
                       verbose=True, shuffle=False)


result = model.evaluate(X_test, y_test)
print(result)
