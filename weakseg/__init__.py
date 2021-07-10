import os
import re
import logging
from datetime import datetime

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.join(PACKAGE_DIR, 'data')
LOGS_LOCATION = os.path.join(PACKAGE_DIR, 'logs')

def _init_logger(log_level_file=logging.INFO, log_level_stream=logging.WARNING):

    logger = logging.getLogger()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s-%(name)s:%(lineno)s-%(levelname)s-%(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    log_name = datetime.now().strftime("log_%Y%m%dT%H%M%S.log")
    log_filepath = os.path.join(LOGS_LOCATION, log_name)
    os.makedirs(LOGS_LOCATION, exist_ok=True)
    
    # general level
    logger.setLevel(log_level_file)
    
    # create file handler with a small log level
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(log_level_file)
    fh.setFormatter(formatter)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level_stream)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logging.info(f"Created logger (log_level_stream: {log_level_stream}, log_level_stream: {log_level_stream}")


_init_logger()