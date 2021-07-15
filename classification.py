from utilities.plotting import generate_3d_plot, generate_2d_plot
from sklearn.cluster import KMeans

import logging
import inspect
import traceback
from utilities.logging import custom_file_handler


file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def kmeans_analyis(
        prediction_output, neurons, x_coord, y_coord,
        init_funnction, n_clusters, n_init, max_iter,
        algorithm='elkan',
        precompute_distances=None,
        random_state=None,
        z_coord=None):
    # Setting up KMeans object
    kmeans = KMeans(
        init=init_funnction,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        precompute_distances=precompute_distances,
        algorithm=algorithm
    )
    kmeans.fit(prediction_output)

    logger.info(
        'Inertia of predicted data based on K-Means is {0}'.format(kmeans.inertia_))
    logger.info(
        'Cluster centers predicted data based on K-Means is {0}'.format(kmeans.cluster_centers_))
    logger.info('Iterations required {0}'.format(kmeans.n_iter_))
    logger.info('KMEANS LABELS {0}'.format(kmeans.labels_))

    if(neurons == 2):
        generate_2d_plot(x_coord, y_coord, kmeans.labels_, True)
    elif(neurons == 3):
        generate_3d_plot(x_coord, y_coord, z_coord, kmeans.labels_, True)
    else:
        logger.info('KMEANS graph cannot be generated. Neuron count is above 3')

    return kmeans.labels_
