import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn
import CNN3
import tensorflow as tf
from LPPL_fitting import LPPL_fit
from data_handler import load_data, prepare_data, get_output_dir
from print_predictions import print_predictions
from dataframe_loader import load_directories, load_4_dataframes, load_dataframes_diff_sizes

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
        
    directories = load_directories(['testowe'])
    dataframes = load_dataframes_diff_sizes(paths=["data\\df_noise.csv"], data_sizes=[400])
    dataframes2 = load_4_dataframes(500)
    

    for df, directory in zip(dataframes, directories):
        print('data loaded')
        output_directory = get_output_dir()+directory+'\\'

        x_train, y_train, x_test, y_test,  y_true, input_shape, nb_classes = prepare_data(df)
        classifier = CNN3.Classifier_CNN(output_directory, input_shape, nb_classes, verbose=False)
        classifier.fit(x_train, y_train, x_test, y_test, y_true)
        prediction = classifier.predict(x_test, y_true, x_train, y_train, y_test)
        print('PREDICTION:', prediction)

        for _df in dataframes2:
            _x_train, _y_train, _x_test, _y_test, _y_true, _input_shape, _nb_classes = prepare_data(_df)
            _prediction = classifier.predict(_x_test, _y_true, _x_train, _y_train, _y_test)
            print('PREDICTION inner:', _prediction)
        
        # print_predictions(classifier)

