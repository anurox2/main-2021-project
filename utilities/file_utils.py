import pandas as pd
from datetime import datetime
from tensorflow.keras import losses

import logging
import inspect
import traceback
from .logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)



def read_data(filename):
    """This function takes in a filename and returns a dataframe
    if the file is a CSV.

    :param filename: Name of CSV file
    :return: dataframe of CSV file
    """
    logger.info('{0} is the name of file to be loaded.'.format(filename))
    if(filename[-3:] == 'csv'):
        df_loaded = pd.read_csv(filename)
        logger.info('{0} has the following shape {1}'.format(filename, df_loaded.shape))
        return df_loaded
    else:
        logger.error(
            'The file is not in CSV format. Please enter a CSV and try again.')
        exit(0)


def get_cwd_filename(loss_function, neurons, scaled, scale_range, batch_size, epochs):
    """This function generates a folder name which will store all the 
    images, plots, and datasets generated while running the code

    :param loss_function: Name of loss function in either string or loss function itself
    :param neurons: Number of neurons in int
    :param scaled: Boolean value if data is scaled or not
    :param scale_range: Tuple for scaling range
    :param batch_size: Batch size for training in int
    :param epochs: Number of epochs to run in int
    :return: Name of directory for this run
    """
    # Rename loss function incase not in str
    if(loss_function == losses.mean_squared_logarithmic_error):
        loss_function = str(loss_function)[10:40]

    # Fetch folder location
    parent_dir = '/home/aman/code_files/main-2021-project/RUNS'
    current_datetime = str(datetime.now().strftime('%d%m%Y-%H%M%S'))

    # Create directory name
    if(scaled):
        dir_name = parent_dir + \
            '/' + current_datetime + \
            '_aman-laptop_' + \
            loss_function + \
            '_N_' + str(neurons) + \
            '_scaled_' + str(scale_range[0]) + '-' + str(scale_range[1]) + \
            '_Epochs_' + str(epochs) + \
            '_batch_' + str(batch_size)
    else:
        dir_name = parent_dir + '/' + \
            current_datetime + \
            '_aman-laptop_' + \
            loss_function + \
            '_N_' + str(neurons) + \
            '_Epochs_' + str(epochs) + \
            '_batch_' + str(batch_size)
    logger.info('Directory name is "{0}"'.format(dir_name))

    return dir_name
