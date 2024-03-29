import tensorflow
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.python.ops.init_ops_v2 import glorot_uniform

from utils.utils import save_logs
from utils.utils import calculate_metrics


class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

        return

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60:
            padding = 'same'

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=input_shape))

        model.add(keras.layers.Conv1D(16, 64, strides=1, padding="same", input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling1D(nb_classes, strides=2, padding="valid"))

        model.add(keras.layers.Conv1D(32, nb_classes, strides=1, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling1D(nb_classes, strides=2, padding="valid"))

        model.add(keras.layers.Conv1D(64, nb_classes, strides=1, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling1D(2, strides=2, padding="valid"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv1D(64, nb_classes, strides=1, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling1D(nb_classes, strides=2, padding="valid"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv1D(64, nb_classes, strides=1, padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Dense(200))
        model.add(keras.layers.Activation('relu'))

        model.add(keras.layers.Dense(nb_classes))
        model.add(keras.layers.Activation('softmax'))

        opt = keras.optimizers.SGD(learning_rate=0.01)

        model.compile(loss=keras.losses.KLDivergence(), optimizer=opt, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        self.callbacks = [model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training
        mini_batch_size = 10
        nb_epochs = 225 #225 jest git ale na próbe można 80 albo mniej bo szybciej

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks,
                                use_multiprocessing=True)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

        keras.backend.clear_session()

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred

    def predicting(self, x_test):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        return y_pred

    #  Not working yet
    def visualize_filter(self, x_train, y_train):

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        model = self.model

        # filters
        filters = model.layers[1].get_weights()[0]
        print(filters)

        new_input_layer = model.inputs
        new_output_layer = [model.layers[1].output]

        new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

        classes = np.unique(y_train)

        colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255)]
        colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

        idx = 10
        idx_filter = 1

        filter = filters[idx_filter]

        plt.figure(1)
        plt.plot(filter + 0.5, color='gray', label='filter')
        for c in classes:
            c_x_train = x_train[np.where(y_train == c)]
            convolved_filter_1 = new_feed_forward([c_x_train])[0]

            idx_c = int(c) - 1

            plt.plot(c_x_train[idx], color=colors[idx_c], label='class' + str(idx_c) + '-raw')
            plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[idx_c],
                     label='class' + str(idx_c) + '-conv')
            plt.legend()
            plt.show()

        # plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')
        return 1



