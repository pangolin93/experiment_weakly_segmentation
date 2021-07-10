import os
import numpy as np

from tqdm.contrib.concurrent import process_map  # or thread_map
from weakseg import DICT_COLOR_INDEX


import logging
logger = logging.getLogger(__name__)


def _unique_void_view(a):
        # https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        return (
                np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))))
                .view(a.dtype)
                .reshape(-1, a.shape[1])
        )
        
def elaborate_single_crop(x):
    
    x = x.reshape((-1, 3))

    list_color = _unique_void_view(x)

    list_cls = [0] * len(DICT_COLOR_INDEX.keys())

    for c in list_color:
        c = tuple(c)
        i = DICT_COLOR_INDEX[c]
        list_cls[i] = 1
    
    return list_cls

def extract_cls(crops_labels, name_set, num_processes=2, flag_save=True):

    list_img_cls = process_map(elaborate_single_crop, crops_labels, max_workers=num_processes, chunksize=20) 

    array_cls_labels = np.array(list_img_cls).astype(int)
    logger.debug(f'array_cls_labels: {array_cls_labels}')

    filepath_txt = os.path.join('data', f'{name_set}_clslabels', 'cls_labels.txt')

    os.makedirs(os.path.dirname(filepath_txt), exist_ok=True)

    if flag_save:
        logger.debug(f'saving array_cls_labels in filepath_txt {filepath_txt}')
        np.savetxt(filepath_txt, array_cls_labels, fmt='%i')


if __name__ == '__main__':

    a = np.ones((3,3,3)) * 255
    a[0][1] = (255, 0, 0)
    a[2][0] = (0, 255, 0)

    b = np.ones((3,3,3)) * 255
    b[0][0] = (0, 255, 255)
    b[1][0] = (0, 255, 255)

    crops_labels = [a, b]

    name_set = 'stupid_test'

    extract_cls(crops_labels, name_set, num_processes=2, flag_save=False)