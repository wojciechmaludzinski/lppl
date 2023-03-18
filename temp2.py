import tensorflow as tf
from tensorflow import keras
import os
PATH = os.getcwd()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU index you want to use


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")

print(tf.config.list_physical_devices())

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


'''
# Define a simple model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate some dummy data
import numpy as np
x_train = np.random.random((100000, 10))
y_train = np.random.randint(2, size=(100000, 1))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
'''
