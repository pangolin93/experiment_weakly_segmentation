
import os
import numpy as np
import torch
import segmentation_models_pytorch as smp


from weakseg.ml.model_segmentation import get_segm_model_and_preprocess_fn

from torch.utils.data.dataloader import DataLoader
from weakseg.ml.custom_augmentation import get_preprocessing, get_validation_augmentation
from weakseg import DATA_DIR, DEVICE
from weakseg.ml.custom_dataset import Dataset
from weakseg.utils.utils_plot import visualize
from weakseg.ml.transform_mask import from_multiclass_mask_to_rgb

import logging
logger = logging.getLogger(__name__)


def test_fn():

    
    x_valid_dir = os.path.join(DATA_DIR, 'val_images')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')


    x_valid_dir = os.path.join(DATA_DIR, 'train_images')
    y_valid_dir = os.path.join(DATA_DIR, 'train_labels')


    # create test dataset
    x_test_dir = x_valid_dir
    y_test_dir = y_valid_dir

    _, preprocessing_fn = get_segm_model_and_preprocess_fn()
    
    # load best saved checkpoint
    best_model = torch.load('./best_model.pth')

    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    test_dataloader = DataLoader(test_dataset, batch_size=32)

    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Accuracy(), 
    ]

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)

    n = np.random.choice(len(test_dataset))
        
    image, gt_mask = test_dataset[n]
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())


    for i in range(5):
        n = np.random.choice(len(test_dataset))
        
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
        dict_images = {
            'image': image.transpose(2, 1, 0), # (3, 224, 224) --> (224, 224, 3)
            'gt_rgb': from_multiclass_mask_to_rgb(gt_mask.transpose(2, 1, 0)).astype(int),
            'rgb_mask': from_multiclass_mask_to_rgb(pr_mask.transpose(2, 1, 0)).astype(int)
        }
    
        visualize(
            images=dict_images,
            save_flag=True,
            filepath_fig=os.path.join('tmp', f'aaa_{i}.png')
        )
    
    return 

if __name__ == '__main__':

    test_fn()