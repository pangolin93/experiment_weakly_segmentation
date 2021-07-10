import os
from weakseg.data_preparation.data_split import get_train_weak_val_indexes
from weakseg.data_preparation.create_clean_dataset import from_raw_to_clean

import logging
logger = logging.getLogger(__name__)

def generate_dataset(num_tot = 33, num_train = 3, num_weak = 23, num_test = 7):

    indexes_train, indexes_weak, indexes_val = get_train_weak_val_indexes(num_tot, num_train, num_weak, num_test)

    folder_images = os.path.join('data_raw', 'images', 'top')
    folder_labels = os.path.join('data_raw', 'labels')

    name_set = 'train'
    indexes = indexes_train
    from_raw_to_clean(folder_images, folder_labels, indexes, name_set)

    name_set = 'weak'
    indexes = indexes_weak
    from_raw_to_clean(folder_images, folder_labels, indexes, name_set)

    name_set = 'val'
    indexes = indexes_val
    from_raw_to_clean(folder_images, folder_labels, indexes, name_set)

    return 

if __name__ == '__main__':
    
    generate_dataset()