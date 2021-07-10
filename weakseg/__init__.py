import os
import logging
from weakseg.general_utils import init_logger

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.join(PACKAGE_DIR, 'data')
LOGS_LOCATION = os.path.join(os.path.dirname(PACKAGE_DIR), 'logs')

# mapping with colors i wanna consider
# no red here
# NOTE: colors are in BGR!
DICT_COLOR_CLS = {
    (255, 255, 255):  0,
    (255,   0,   0):  1,
    (255, 255,   0):  2,
    (0, 255,   0):    3,
    (0, 255, 255):    4,
} 

RANDOM_STATE = 21

init_logger(log_dir=LOGS_LOCATION, log_level_file=logging.DEBUG, log_level_stream=logging.WARNING)