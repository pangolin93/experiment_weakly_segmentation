import os
from weakseg.ml.custom_train_step import WeaklyTrainEpoch, WeaklyValidEpoch
from weakseg.ml.weakly.image_classifier import ImageClassifier

import torch
from weakseg.ml.balance_classes import get_balancer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import segmentation_models_pytorch as smp

from weakseg.ml.custom_dataset import Dataset
from weakseg.ml.model_segmentation import get_segm_model_and_preprocess_fn
from weakseg.ml.custom_augmentation import get_preprocessing, get_training_augmentation, get_validation_augmentation
from weakseg import DATA_DIR

import logging
logger = logging.getLogger(__name__)


def train_fn(filepath_best_model='best_model.pth'):

    # https://github.com/qubvel/segmentation_models.pytorch/issues/265
    img_size = 224

    model, preprocessing_fn = get_segm_model_and_preprocess_fn()

    x_train_dir = os.path.join(DATA_DIR, 'train_images')
    y_train_dir = os.path.join(DATA_DIR, 'train_labels')

    x_valid_dir = os.path.join(DATA_DIR, 'val_images')
    y_valid_dir = os.path.join(DATA_DIR, 'val_labels')


    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(img_size=img_size), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(img_size=img_size), 
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    list_mask = [train_dataset[i][1] for i in tqdm(range(train_dataset.num_images))]
    weighted_sampler, avg_perc_classes = get_balancer(list_mask)

    train_loader = DataLoader(train_dataset, batch_size=8*2, num_workers=2, sampler=weighted_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=8*2, shuffle=False, num_workers=2)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    fn_loss_strong = smp.utils.losses.DiceLoss()

    fn_loss_weak = torch.nn.MSELoss() # (weight=torch.from_numpy(1/avg_perc_classes))

    metrics_strong = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Accuracy(), 
    ]

    metrics_weak = [
        smp.utils.metrics.Accuracy(), 
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    DEVICE = 'cuda'
    model = model.to(DEVICE)

    model_weak = ImageClassifier()
    model_weak = model_weak.to(DEVICE)

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = WeaklyTrainEpoch(
        model=model, 
        loss_strong=fn_loss_strong, 
        loss_weak=fn_loss_weak, 
        metrics_strong=metrics_strong, 
        metrics_weak=metrics_weak, 
        optimizer=optimizer, 
        device=DEVICE, 
        enable_weak=True, 
    )

    valid_epoch = WeaklyValidEpoch(
        model=model, 
        loss_strong=fn_loss_strong, 
        loss_weak=fn_loss_weak, 
        metrics_strong=metrics_strong, 
        metrics_weak=metrics_weak, 
        device=DEVICE, 
        enable_weak=False, 
    )

    max_score = 0
    for i in range(0, 15):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['fscore']:
            max_score = valid_logs['fscore']
            torch.save(model, filepath_best_model)
            print('Model saved!')
            
    return

if __name__ == '__main__':

    train_fn()

