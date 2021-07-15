
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import logging
import inspect
import traceback
from .logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def plotcolor(coord_list):
    colors = []
    for item in coord_list:
        if item == 1:
            colors.append('red')
        else:
            colors.append('green')
    logger.info(
        'Color list generated based on values. 1 is for RED, 0 is for GREEN'
    )
    return colors


def generate_2d_plot(x_coord, y_coord, palette_colors, kmeans_bool=False):
    """This function generates a 2-dimensional figure in the folder structure"""

    filename = 'prediction_2D.png'
    if(kmeans_bool):
        filename = 'kmeans_prediction_2D.png'
    fig_2d, ax_2d = plt.subplots(1)
    ax_2d.set_xlabel('X Label - Pred')
    ax_2d.set_ylabel('Y Label - Pred')

    ax_2d.scatter(x_coord, y_coord, c=palette_colors)

    fig_2d.savefig(filename)
    logger.info('2D figure generated')
    del fig_2d, ax_2d


def generate_3d_plot(x_coord, y_coord, z_coord, plt_colors, kmeans_bool=False):
    filename = 'prediction_3D.png'
    if(kmeans_bool):
        filename = 'kmeans_prediction_3D.png'

    try:
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_coord, y_coord, z_coord, c=plt_colors)

        ax.set_xlabel('X Label - Pred')
        ax.set_ylabel('Y Label - Pred')
        ax.set_zlabel('Z Label - Pred')

        plt.savefig(filename)
        logger.info('3D figure generated')
        del ax, fig

    except Exception:
        logger.exception('3D figure generation failed {0}'.format(Exception))


def generate_training_plots(history):
    try:
        training_fig, training_ax = plt.subplots(1)
        training_ax.plot(history.history['loss'], label='train')
        training_ax.plot(history.history['val_loss'], label='test')
        training_ax.legend()
        training_fig.savefig("training.png")
        logger.info('Training graphs generated')
    except Exception:
        logger.exception(
            'Traning graphs could not be generated. Exception: {0}'.format(
                Exception
            ))


def generate_3d_movable_graph(x_coord, y_coord, z_coord, color,
                              opacity=0.6,
                              color_continuous_scale='bluered',
                              kmeans_bool=False):
    filename = '3D_graph_movable.html'
    if(kmeans_bool):
        filename = '3D_graph_movable_KMeans_labels.html'
    movable_3d_figure = px.scatter_3d(
        x=x_coord,
        y=y_coord,
        z=z_coord,
        color=color,
        opacity=opacity,
        color_continuous_scale=color_continuous_scale
    )
    
    movable_3d_figure.write_html(filename)
    logger.info('3D movable graph generated')
    del movable_3d_figure
