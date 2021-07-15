# -*- coding: utf-8 -*-
import os

from tensorflow.keras import losses

from utilities.file_utils import get_cwd_filename, read_data
from utilities.data_processing import process_data, data_scaling
from utilities.autoencoder_design import design
from utilities.plotting import plotcolor, generate_2d_plot, generate_3d_plot, generate_training_plots, generate_3d_movable_graph
from prediction_data_processing import save_pred_data
from utilities.confusion_matrix import calculate_and_generate_confusion_matrix
from classification import kmeans_analyis
from utilities.plotting import generate_training_plots

import logging
import inspect
import traceback
from utilities.logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

# Setting up variables for run
randomized_data_df = read_data('data/randomized_doh_data.csv')
randomized_data_df = randomized_data_df.head(10000)
epochs = 10
batch_size = 32
scaled = True
scale_range = (0, 100)
neurons = 3

design_structures = ['24_8_3', '24_12_6_3', '24_18_12_8_6_3', ]
design_number = 1

loss_function_list = [
    'mse', losses.mean_squared_logarithmic_error, 'binary_crossentropy']

loss_function_number = 1
loss_function = loss_function_list[loss_function_number-1]

# Variables for classification
clusters = 2
ranmdom_state = None
logger.info(
    '---------------------------------------------------------------------------------------------')
logger.info('''STARTING EXECUTION
                                            Loss function - {0}
                                            Neurons - {1}
                                            Scaled - {2}
                                            Scale Range - {3}
                                            Batch Size  - {4}
                                            Epochs - {5}'''.format(
    str(loss_function),
    str(neurons),
    str(scaled),
    str(scale_range),
    str(batch_size),
    str(epochs),
))

dir_name = get_cwd_filename(
    loss_function,
    neurons,
    scaled,
    scale_range,
    batch_size,
    epochs
)
print(dir_name)
logger.info('{0} is the current working directory'.format(os.getcwd()))
try:
    os.mkdir(dir_name)
except Exception:
    logger.error(
        'EXITING CODE. Could not create directory to store deliverables')
    logger.exception(Exception)
    print('EXITING CODE. Could not create directory to store deliverables')
    exit(100)

os.chdir(dir_name)
logger.info('Directory changed to {0}'.format(os.getcwd()))
# ----------------------------------------------------------

X, y, X_train, X_test, y_train, y_test = process_data(randomized_data_df)

# Scaling data based on option
if(scaled):
    X_train, X_test = data_scaling(X, X_train, X_test, scale_range)

# for loss_function in loss_function_list:
logger.info("Starting autoencoder process for loss function {0}".format(loss_function))
model, encoder_no_decoder = design(
    design_structures[design_number-1], X, loss_function, 'linear')
logger.info("Autoencoder and encoder models built")

logger.info("Training process started")
history = model.fit(X_train,
                    X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(X_test, X_test)
                    )
logger.info("Training process completed")

# Plotting results
try:
    logger.info("The accuracy is", history.history['accuracy'])
    print("The accuracy is", history.history['accuracy'])
except Exception:
    logger.exception(
        'Accuracy could NOT be extracted. Exception:{0}'.format(Exception))
    print('Accuracy could NOT be extracted. Check log file for more details')

generate_training_plots(history)

# PREDICTION
prediction_output = encoder_no_decoder.predict(X_test)
logger.info('Prediction completed. Predicted and ACTUAL values in console')
print("######## PREDICTION values ########")
print(prediction_output)

print("######## ACTUAL values ############")
print(y_test)

save_pred_data(neurons, prediction_output)
plt_colors = plotcolor(y_test)

# Extracting x, y and z coordinates for plotting.
x_coord = []
y_coord = []
for i in range(0, len(prediction_output)):
    x_coord.append(prediction_output[i][0])
    y_coord.append(prediction_output[i][1])

z_coord = None

if(neurons == 2):
    logger.info('Printing 2D graph')
    generate_2d_plot(x_coord, y_coord, plt_colors)
    logger.info('Graph printed')

elif(neurons == 3):
    z_coord = []
    for i in range(0, len(prediction_output)):
        z_coord.append(prediction_output[i][2])
    logger.info('Printing 3D graph')
    generate_3d_plot(x_coord, y_coord, z_coord, plt_colors)
    logger.info('Graph printed')

else:
    logger.warn(
        'Too many neurons. Saving prediction output for further analysis')
    print('Too many neurons. Saving prediction output for further analysis')

kmeans_labels = kmeans_analyis(
    prediction_output,
    neurons,
    x_coord,
    y_coord,
    init_funnction='k-means++',
    n_clusters=clusters,
    n_init=10,
    max_iter=300,
    algorithm='elkan',
    precompute_distances=True,
    random_state=None,
    z_coord=z_coord
)

if(z_coord is not None):
    generate_3d_movable_graph(
        x_coord, y_coord, z_coord, kmeans_labels, 0.6, 'bluered', True)

calculate_and_generate_confusion_matrix(y_test, kmeans_labels)
