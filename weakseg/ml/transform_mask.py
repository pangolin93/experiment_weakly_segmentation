import numpy as np
from weakseg import DICT_INDEX_COLOR

import logging
logger = logging.getLogger(__name__)

def from_multiclass_mask_to_rgb(mask, classes):
    # mask.shape = (width, heigth, n_classes) --> (224, 224, 5)
    
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3))     

    for i in DICT_INDEX_COLOR:
        m_tmp = np.take(mask>0, i, axis=2) 
        rgb_image[m_tmp] = DICT_INDEX_COLOR[i]

    return rgb_image