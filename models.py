'''
Author: Ambareesh Ravi
Date: July 19, 2021
Title: models.py
Project: For InvisionAI recuritment
Description:
    Contains the PyTorch CNN model for rooftop pixel classification/segmentation
'''

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch model for Pixelwise rooftop classifcation
class PixelClassifier_CNN(nn.Module):
    def __init__(self, in_channels = 3):
        '''
        Overrides nn.Module to create a PyTorch model for classifying pixels with roofs in buildings
        
        Args:
            in_channels - number of input channels as <int> [3 for RGB, 1 for Grayscale]
            
        Returns:
            -
        
        Exception:
            -
        '''
        super().__init__()
        
        # Class variables
        self.in_channels = in_channels
        
        # Given Layers
        self.conv1 = nn.Conv2d(in_channels = self.in_channels, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.tr_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, padding = 0)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 0)
        self.max_pool = nn.MaxPool2d(2, 2)
        
        # Extra layers
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride = 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, padding=1),
            nn.Sigmoid()
        )
                
    def forward(self, input_images):
        '''
        Processes the input images through the CNN
        
        Args:
            input_images - batch of input images as <torch.Tensor>
            
        Returns:
            output_masks - grayscale mask containing white pixels for roofs and black pixels for others as <torch.Tensor>
        
        Exception:
            -
        '''
        conv1_out = F.relu(self.conv1(input_images))
        conv2_out = F.relu(self.conv2(conv1_out))
        max_pool1_out = self.max_pool(conv2_out)
        conv3_out = F.relu(self.conv3(max_pool1_out))
        tr_conv1_out = self.tr_conv1(conv3_out)
        conv4_out = self.conv4(tr_conv1_out)
        output_masks = self.output_layer(conv4_out)
        return output_masks