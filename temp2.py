import tensorflow as tf
import os
PATH = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

if tf.config.list_physical_devices():
    print("GPU is available")
else:
    print("GPU is not available")

    import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU index you want to use

import tensorflow as tf

# Define a simple model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate some dummy data
import numpy as np
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)