import os
from weakseg.data_preparation.extract_cls_label import extract_cls
from weakseg.data_preparation.utils_preprocessing import elaborate_images, remove_red

import logging
logger = logging.getLogger(__name__)

def from_raw_to_clean(folder_images, folder_labels, indexes, name_set):

    logger.info(f'preparing {name_set} set ... ')

    ################################################################################################
    # crop images 

    dst_folder=os.path.join('data', f'{name_set}_images')
    logger.info(f'elaborate_images INPUT DATA saving into {dst_folder}... ')
    crops_images = elaborate_images(folder_data=folder_images, dst_folder=dst_folder, indexes=indexes, funct_to_apply=None)

    ################################################################################################
    # crop labels + remove red
    
    dst_folder=os.path.join('data', f'{name_set}_labels')
    logger.info(f'elaborate_images LABEL DATA saving into {dst_folder}... ')
    crops_labels = elaborate_images(folder_data=folder_labels, dst_folder=dst_folder, indexes=indexes, funct_to_apply=remove_red)

    ################################################################################################
    # save cls labels

    logger.info(f'extracting cls from mask labels')
    extract_cls(crops_labels, name_set)

if __name__ == '__main__':
    folder_images = os.path.join('data_raw', 'images', 'top')
    folder_labels = os.path.join('data_raw', 'labels')

    indexes = [0,1,2]
    name_set = 'stupid_test'

    from_raw_to_clean(folder_images, folder_labels, indexes, name_set)