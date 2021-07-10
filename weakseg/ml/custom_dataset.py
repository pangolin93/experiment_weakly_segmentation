import os
import cv2
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from weakseg import DATA_DIR, DICT_COLOR_INDEX

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """

    # NOTE: opencv opens BFR by default!
    
    DICT_COLOR_INDEX = DICT_COLOR_INDEX

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):

        self.ids = os.listdir(images_dir)

        self.num_images = len(self.ids)
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.class_values = self.DICT_COLOR_INDEX
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(str(self.images_fps[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_mask = cv2.imread(str(self.masks_fps[i]))
        original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2RGB)

        # extract certain classes from mask
        masks = [(np.all(original_mask == k, axis=-1)) for k in self.DICT_COLOR_INDEX]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)


if __name__ == '__main__':


    x_train_dir = os.path.join(DATA_DIR, 'train_images')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')

    x_valid_dir = os.path.join(DATA_DIR, 'val_images')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

    dataset = Dataset(x_train_dir, y_train_dir)

    image, mask = dataset[42] # get some sample

    print((image.shape, mask.shape))