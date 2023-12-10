# in this data file, define a dataset of random data with only one example, we are trying to overfit the model

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

class RandomDataset(Dataset):
    def __init__(self, channels:int=96, circle=False, step:int=8):
        '''
            channels: the number of channels in the input image
            step: the size of the checkerboard square
        '''
        self.channels = channels
        self.input = torch.randn(channels, 64, 64)
        if not circle:
            self.label = torch.zeros(1, 64, 64)
            self.step = step
            for i in range(0, 64, self.step):
                for j in range(0, 64, self.step):
                    if (i // self.step) % 2 == (j // self.step) % 2:
                        self.label[:, i:i+self.step, j:j+self.step] = 1
        else:
            # read "lable.png" as the label in 1 channel
            self.label = cv2.imread('label.png', cv2.IMREAD_GRAYSCALE)
            self.label = torch.from_numpy(self.label).unsqueeze(0).float()
            
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.input, self.label
    
    
if __name__ == '__main__':
    random_dataset = RandomDataset()
    x, y = random_dataset[0]
    # select one channel and plot it, the images are grayscale images
    plt.imshow(x[0], cmap='gray')
    plt.show()
    plt.imshow(y.permute(1, 2, 0), cmap='gray')
    plt.show()
    
    
            