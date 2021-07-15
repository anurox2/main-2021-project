import logging
import logging.handlers
from datetime import datetime

def custom_file_handler():
    log_file_name = '{0}_.log'.format(str(datetime.now().strftime('%Y_%m_%d')))

    f_handler = logging.FileHandler(log_file_name, mode='a')
    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    f_handler.setFormatter(f_format)
    return f_handler