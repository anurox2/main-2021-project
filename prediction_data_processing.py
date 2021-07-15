from utilities.plotting import plotcolor, generate_2d_plot, generate_3d_plot
import pandas as pd

import logging
import inspect
import traceback
from utilities.logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

def save_pred_data(neurons, prediction_output):
    neuron_header_list = []
    for neuron in range(0, neurons):
        neuron_header_list.append('n' + str(neuron + 1))

    prediction_data = pd.DataFrame(
        prediction_output,
        columns=neuron_header_list
        )

    prediction_data.to_csv('predicted_data.csv', index=False)
    logger.info('Prediction data saved to CSV file')