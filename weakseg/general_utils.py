import logging
import os
from datetime import datetime

def init_logger(log_dir: str, log_level_file=logging.INFO, log_level_stream=logging.WARNING):

    logger = logging.getLogger()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s-%(name)s:%(lineno)s-%(levelname)s-%(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    log_name = datetime.now().strftime("log_%Y%m%dT%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_name)
    os.makedirs(log_dir, exist_ok=True)
    
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

