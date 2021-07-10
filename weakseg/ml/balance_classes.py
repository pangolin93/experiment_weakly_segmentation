import numpy as np
from torch.utils.data import WeightedRandomSampler

def make_weights_for_balanced_classes(masks, nclasses):                        
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    
    count_classes = [0] * nclasses

    num_mask = len(masks)

    # masks --> List[mask]
    # mask.shape --> (5, 224, 224)

    list_perc = []
    for mask in masks:
      # for each image i compute % for each class                                                         
      perc_classes_tmp = (
          mask.sum(axis=1).sum(axis=1) / (mask.shape[0]*mask.shape[1])
      )
      list_perc.append(perc_classes_tmp)

      count_classes += perc_classes_tmp

    # avoid dividing by 0 or very small number
    count_classes += 0.05
    sum_count_classes = sum(count_classes)
    perc_classes = count_classes / sum_count_classes

    weight_classes = 1 / perc_classes

    weights = [0] * num_mask
    for i in range(num_mask):                                                     
      weights[i] = np.dot(list_perc[i], weight_classes)

    return weights, perc_classes    

def get_balancer(list_mask, nclasses=5):

    # https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#WeightedRandomSampler

    nclasses=5
    num_samples = len(list_mask)
    weights, perc_classes = make_weights_for_balanced_classes(list_mask, nclasses=nclasses)
    weighted_sampler = WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)

    return weighted_sampler, perc_classes
