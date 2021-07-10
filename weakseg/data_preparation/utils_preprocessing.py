import itertools
import cv2
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from weakseg import DICT_COLOR_INDEX

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def remove_red(img):

    img = img.astype(int)

    # find the only red pixels with the mask 
    # axis = 2 since channel 
    # NOTE: open cv decode B G R
    # https://docs.opencv.org/4.5.2/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    mask = (img == [0,0,255]).all(axis=2)

    # apply the mask to overwrite the pixels
    img[ mask ] = [255,255,255]

    return img

def create_list_sub_img(images, stepSize=100, windowSize=200):

    # https://stackoverflow.com/questions/61051120/sliding-window-on-a-python-image
    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                res = image[y:y + windowSize[1], x:x + windowSize[0]]
                # i keep only images with this size!
                if res.shape == (windowSize[0], windowSize[1], 3):
                    yield res

    x = [sliding_window(image, stepSize, (windowSize, windowSize)) for image in images]

    crops = list(itertools.chain.from_iterable(x))

    return crops


def elaborate_images(folder_data, dst_folder, indexes, funct_to_apply=None):
    list_filepath_images = [os.path.join(folder_data, f) for f in os.listdir(folder_data) if os.path.isfile(os.path.join(folder_data, f))]

    # needed to split in train, weak, test
    list_filepath_images = [list_filepath_images[i] for i in indexes]

    list_images = [cv2.imread(filepath, cv2.IMREAD_UNCHANGED) for filepath in list_filepath_images]
    
    if funct_to_apply is not None:
        list_images = [funct_to_apply(x) for x in list_images]

    crops_images = create_list_sub_img(list_images)

    shutil.rmtree(dst_folder, ignore_errors=True)

    os.makedirs(dst_folder, exist_ok=True)

    for i in tqdm(range(len(crops_images))):
        img = crops_images[i]
        dst_filepath = os.path.join(dst_folder, f'{i}.tif')
        cv2.imwrite(dst_filepath, img)

    return crops_images

