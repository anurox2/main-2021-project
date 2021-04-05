# -*- coding: utf-8 -*-
"""
Master's graduation project. The aim is to detect DoH vs HTTPS traffic.

Written by: Aman Kumar Gupta (X397J446)

Faculty Advisor: Dr. Sergio Salinas Monroy
"""

# Data manipulation libraries
import pandas as pd
from numpy import array
import numpy as np
import os
from datetime import datetime

# train autoencoder for classification with no compression in the bottleneck layer
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras import losses
from keras.optimizers import Adam

# Plotting
# from matplotlib import pyplot
import matplotlib.pyplot as plt

# Loading saved model


# Verifying the cluster
# from kneed import KneeLocator
# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# Another clustering
# from mpl_toolkits.mplot3d import Axes3D

"""---
## Loading data ---------------------------------------------------------------------------------------------------------------------------
"""
os.chdir('/home/x397j446/main-project')


def read_data(filename):
  df_loaded = pd.read_csv(filename)
  print('', filename, 'has the following shape', df_loaded.shape)
  return df_loaded


def get_cwd_filename(loss_function, neurons, scaled, scale_range, batch_size, epochs):
  # Rename loss function incase not in str
  if(loss_function == losses.mean_squared_logarithmic_error):
    loss_function = str(loss_function)[10:40]
  
  # Fetch folder location
  parent_dir = '/home/x397j446/main-project'
  current_datetime = str(datetime.now().strftime('%d%m%Y-%H%M%S'))
  
  # Create directory name
  if(scaled):
    dir_name = parent_dir + '/' + current_datetime + '_HPC_RANDOM_' + loss_function + '_N_' + str(neurons) + '_scaled_' + str(scale_range[0]) + '-' + str(scale_range[1]) + '_Epochs_' + str(epochs)
  else:
    dir_name = parent_dir + '/' + current_datetime + '_HPC_RANDOM_' + loss_function + '_N_' + str(neurons) + '_Epochs_' + str(epochs)
  
  return dir_name


def process_data(dataframe_raw):
  # Extract X and y values
  X = dataframe_raw.drop(columns=['is_doh']).to_numpy()
  y = dataframe_raw['is_doh'].to_numpy()

  # Split the data into test and train
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

  return X, y, X_train, X_test, y_train, y_test


def data_scaling(X_train, X_test, scale_range):
  scaler = MinMaxScaler(feature_range=scale_range)
  scaler.fit(X_train)
  X_train_scaled = scaler.transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, X_test_scaled


def design_autoencoder(X, neurons, loss_function, activation='linear'):
  # Encoder design
  number_of_inputs = X.shape[1]
  input_encoder = Input(shape=(number_of_inputs,))

  ## Encoder level 1
  encoder_1 = Dense(number_of_inputs*2)(input_encoder)
  encoder_1 = BatchNormalization()(encoder_1)
  encoder_1 = LeakyReLU()(encoder_1)

  ## Encoder level 2
  encoder_2 = Dense(number_of_inputs)(encoder_1)
  encoder_2 = BatchNormalization()(encoder_2)
  encoder_2 = LeakyReLU()(encoder_2)

  divisor = number_of_inputs/neurons

  # Bottleneck design
  bottleneck_inputs = number_of_inputs/divisor
  bottleneck = Dense(bottleneck_inputs)(encoder_2)

  # Decoder design
  ## Decoder Level 1
  decoder_1 = Dense(number_of_inputs)(bottleneck)
  decoder_1 = BatchNormalization()(decoder_1)
  decoder_1 = LeakyReLU()(decoder_1)

  ## Decoder Level 2
  decoder_2 = Dense(number_of_inputs*2)(decoder_1)
  decoder_2 = BatchNormalization()(decoder_2)
  decoder_2 = LeakyReLU()(decoder_2)


  output = Dense(number_of_inputs, activation=activation)(decoder_2)

  # Defining the autoencoder model
  model = Model(inputs=input_encoder, outputs=output)
  model.compile(Adam(0.001), loss=loss_function)
  plot_model(model, "autoencoder.png", show_shapes=True)

  # define an encoder model (without the decoder)
  encoder_no_decoder = Model(inputs=input_encoder, outputs=bottleneck)
  plot_model(encoder_no_decoder, 'encoder.png', show_shapes=True)

  return model, encoder_no_decoder


def plotcolor(coord_list):
  colors = []
  for item in coord_list:
    if item == 1:
      colors.append('red')
    else:
      colors.append('green')
  return colors


def generate_2d_plot(x_coord, y_coord, plt_colors):
  fig_2d, ax_2d = plt.subplots(1)
  
  ax_2d.scatter(x_coord, y_coord, c=plt_colors)

  ax_2d.set_xlabel('X Label - Pred')
  ax_2d.set_ylabel('Y Label - Pred')

  fig_2d.savefig('prediction_2D.png')


def generate_3d_plot(x_coord, y_coord, z_coord, plt_colors):
  try:
    fig = plt.figure()

    ax = fig.add_subplot(111, projection = '3d')

    ax.scatter(x_coord, y_coord, z_coord, c = plt_colors)

    ax.set_xlabel('X Label - Pred')
    ax.set_ylabel('Y Label - Pred')
    ax.set_zlabel('Z Label - Pred')

    plt.savefig('prediction_3D.png')
    return "Success", 'ExceptionNOT'
    
  except Exception:
    return "Error in generate 3d plot function", Exception




"""---
## Select options -------------------------------------------------------------------------------------------------------------------------
"""
epochs = 100
batch_size = 16
scaled = True
scale_range = (0, 100)
neurons = 10

loss_function_list = ['mse', losses.mean_squared_logarithmic_error, 'binary_crossentropy']

# Reading data
print('\n\nReading data')
randomized_data_df = read_data('randomized_doh_data.csv')
print('\n\nData loaded')

# Processing data
print('\n\nProcessing data')
X, y, X_train, X_test, y_train, y_test = process_data(randomized_data_df)
print('\n\nData processed')

# Scaling data based on option
if(scaled):
  X_train, X_test = data_scaling(X_train, X_test, scale_range)

for loss_function in loss_function_list:
  print("\n\nStarting autoencoder process for loss function", loss_function)
  
  # Get the directory to store all files in
  current_dir_name = get_cwd_filename(loss_function, neurons, scaled, scale_range, batch_size, epochs)
  os.mkdir(current_dir_name)
  os.chdir(current_dir_name)

  print("\n\nDirectory created at", current_dir_name)

  model, encoder_no_decoder = design_autoencoder(X, neurons, loss_function, 'linear')
  
  # TRAINING -------------------_-------------------_-------------------_-------------------_-------------------_-------------------_
  history = model.fit(X_train, 
                      X_train,
                      epochs = epochs,
                      batch_size = batch_size,
                      verbose = 1,
                      validation_data = (X_test, X_test)
                      )
            # -------------------_-------------------_-------------------_-------------------_-------------------_-------------------_
  
  # Plotting results
  try:
    print("The accuracy is", history.history['accuracy'])
  except:
    print('Accuracy could NOT be extracted')

  training_fig, training_ax = plt.subplots(1)
  training_ax.plot(history.history['loss'], label='train')
  training_ax.plot(history.history['val_loss'], label='test')
  training_ax.legend()
  training_fig.savefig("training.png")
  


  # PREDICTION -------------------_-------------------_-------------------_-------------------_-------------------_-------------------_
  prediction_output = encoder_no_decoder.predict(X_test)
  print("######## PREDICTION values ########")
  print(prediction_output)

  print("######## ACTUAL values ############")
  print(y_test)

  x_coord = []
  y_coord = []
  for i in range(0, len(prediction_output)):
    x_coord.append(prediction_output[i][0])
    y_coord.append(prediction_output[i][1])

  if(neurons == 2):   
    plt_colors = plotcolor(y_test)
    print('Printing graphs')
    generate_2d_plot(x_coord, y_coord, plt_colors)
    print('Graphs printed')

  elif(neurons == 3):
    z_coord = []
    for i in range(0, len(prediction_output)):
      z_coord.append(prediction_output[i][2])

      # Get the colors
      plt_colors = plotcolor(y_test)
      
      message, exception = generate_3d_plot(x_coord, y_coord, z_coord, plt_colors)
  
  else:
    print('Too many neurons. Saving prediction output for further analysis')
    
  neuron_header_list = []
  for neuron in range(0, neurons):
    neuron_header_list.append('n'+str(neuron+1))
  
  prediction_data = pd.DataFrame(prediction_output, columns = neuron_header_list)
  prediction_data.to_csv('predicted_data.csv', index=False)

  print('\n\n Performing K-Means analysis')
  kmeans = KMeans(
    init='random',
    n_clusters=2,
    n_init=10,
    max_iter=300,
    random_state=None,
  )

  kmeans.fit(prediction_output)
  print("\n\tInertia of predicted data based on K-Means is", kmeans.inertia_)
  print("\n\tCluster centers predicted data based on K-Means is", kmeans.cluster_centers_)
  print("\n\tIterations required", kmeans.n_iter_)
  print('KMEANS LABELS', kmeans.labels_)

  print("\n\t2d graph of K-means clustering")

  fig, ax1 = plt.subplots(
      1, 
  )
  fte_colors = {
      0: 'green',
      1: 'red'
  }
  km_colors = [fte_colors[label] for label in y_test]
  ax1.scatter(
      x_coord, 
      y_coord,
      c=km_colors
  )

  plt.savefig('K-means-clustering.png')