import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()

        # NOTE: this is a very simple classifier 

        in_channel_start = 3
        n_classes = 5

        # 3 --> input channel
        self.conv1 = nn.Conv2d(in_channel_start, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5)
        self.conv3 = nn.Conv2d(8, n_classes, kernel_size=5)

        # i want to classify... or predict % pixels of each class in frame
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x