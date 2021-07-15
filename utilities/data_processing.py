import pandas as pd
import numpy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging
import inspect
import traceback
from .logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def process_data(dataframe_raw):
    """This function returns the dataset in numpy arrays
    for training purposes

    :param dataframe_raw: Dataframe from file
    :return: X, y, X_train, X_test, y_train, y_test in numpy arrays
    """
    # Extract X and y values
    try:
        X = dataframe_raw.drop(columns=['is_doh']).to_numpy()
        y = dataframe_raw['is_doh'].to_numpy()

        logger.info(
            'Splitting data into 70%% for training, and 30%% for test, and no randomization.'
        )
        # Split the data into test and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.30,
            random_state=0
        )
        logger.info('Data splitting successful. Returning numpy arrays')

        return X, y, X_train, X_test, y_train, y_test
    except Exception:
        logger.error('Unable to process data')
        logger.exception(Exception)


def data_scaling(X, X_train, X_test, scale_range):
    """This function scales the date training and testing X data 
    in the scale range provided

    :param X_train: X data for training
    :param X_test: X data for testing
    :param scale_range: Scale range for scaling
    :return: X_train_scaled, X_test_scaled scaled numpy arrays
    """
    scaler = MinMaxScaler(feature_range=scale_range)
    logger.info('Attempting to fit X to the scaler')
    scaler.fit(X)
    logger.info('Fitting complete')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info('Data scaler between {0}-{1} range successfully'.format(
        str(scale_range[0]),
        str(scale_range[1])
    ))
    return X_train_scaled, X_test_scaled
