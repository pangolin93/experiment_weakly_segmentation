
from typing import Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def visualize(images: Dict[str, np.ndarray], save_flag=False, filepath_fig='img.png'):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

    if save_flag:
        logger.info(f'saving figure in {filepath_fig}')
        plt.savefig(filepath_fig)