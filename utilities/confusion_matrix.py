from sklearn.metrics import confusion_matrix

import logging
import inspect
import traceback
from utilities.logging import custom_file_handler

file_handler = custom_file_handler()
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def calculate_and_generate_confusion_matrix(y_test, classification_labels):
    conf_matrix = confusion_matrix(y_test, classification_labels)
    logger.info('Confusion matrix generated')
    print(conf_matrix)

    f = open('confusion_matrix.txt', 'w')
    f.write('CONF_MATRIX' + repr(conf_matrix) + '\n')
    f.close()
    logger.info('Confusion matrix written to file')
