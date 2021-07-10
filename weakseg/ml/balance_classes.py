import numpy as np
from torch.utils.data import WeightedRandomSampler

def make_weights_for_balanced_classes(masks, nclasses):                        
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    
    count_classes = [0] * nclasses

    num_mask = len(masks)
    for mask in masks:
      # for each image i compute % for each class                                                         
      perc_classes_tmp = (
          mask.sum(axis=0).sum(axis=0) / (mask.shape[0]*mask.shape[1])
      )

      count_classes += perc_classes_tmp

    # avoid dividing by 0 or very small number
    count_classes += 0.05
    sum_count_classes = sum(count_classes)
    perc_classes = count_classes / sum_count_classes

    weight_classes = 1 / perc_classes

    weights = [0] * num_mask
    for i in range(num_mask):
      mask = masks[i]
      # for each image i compute % for each class                                                         
      perc_classes_tmp = (
          mask.sum(axis=0).sum(axis=0) / (mask.shape[0]*mask.shape[1])
      )

      weights[i] = np.dot(perc_classes_tmp, weight_classes)

    return weights       

def get_balancer(list_mask, nclasses=5):

    # https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#WeightedRandomSampler

    nclasses=5
    num_samples = len(list_mask)
    weights = make_weights_for_balanced_classes(list_mask, nclasses=nclasses)
    weighted_sampler = WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)

    return weighted_sampler
