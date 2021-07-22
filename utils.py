'''
Author: Ambareesh Ravi
Date: July 18, 2021
Title: utils.py
Project: For InvisionAI recuritment
Description:
    Contains utility and helper functions for the project
'''

# Libraries imports
import numpy as np
import os
from tqdm import tqdm
from time import time
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

# Global variables
MANUAL_SEED = 42
np.random.seed(42)

def INFO(s):
    '''
    Prints information in a particular format

    Args:
        s - string <str> to be printed

    Returns:
        -

    Exception:
        -
    '''
    print("-"*40)
    print("INFO:", s)
    print("-"*40)

def read_directory_content(path):
    '''
    Reads all files in a directory given a path
    
    Args:
        path - path for the directory as <str>
    
    Returns:
        sorted list of files in the directory
    
    Exception:
        -
    '''
    if "*" not in path: path = os.path.join(path, "*")
    return sorted(glob(path))
    
def create_directory(path):
    '''
    Creates a directory given a path if the path does not exist
    
    Args:
        path - path for the directory as <str>
    
    Returns:
        -
    
    Exception:
        -
    '''
    # Create a directory
    if not os.path.exists(path): os.mkdir(path)

def save_image(array, path, resize = False, extension = ".png"):
    '''
    Saves an array into an image file
    
    Args:
        array - image as a <np.array>
        path - path for the image as <str>
        resize - [optional] to resize image to given size - <tuple> of <int> (w,h)
        extension - [optional] type of image file as <str>
    
    Returns:
        -
    
    Exception:
        -
    '''
    # Add image extension
    if extension not in path:
        path = path.split(".")[0] + extension
        
    # Save image into a file using PIL Image handle
    img = Image.fromarray(array)
    # Resize image if reaquired
    if resize: img = img.resize(resize)
    # Save image
    img.save(path)
    
def read_image(image_path):
    '''
    Reads an image from the given path as a PIL.Image handle
    
    Args:
        image_path - path for the image as <str>
    
    Returns:
        -
    
    Exception:
        -
    '''
    return Image.open(image_path)

class Visualizer:
    def __init__(self,):
        '''
        Initializes the class to visualize results in comparison with the inputs
        
        Args:
            -
            
        Returns:
            -
            
        Exception:
            -
        '''
        pass
    
    def gray2color(self, x):
        '''
        Converts a single channel grayscale image to coloured 3 channel format
        
        Args:
            x - input as <np.array>
            
        Returns:
            -
            
        Exception:
            -
        '''
        return np.repeat(np.expand_dims(x, axis = -1), 3, axis = -1)
        
    def visualize_composite(self, input_image, label, prediction, margin = 8, save_path = None):
        '''
        Function to visualize input, label, prediction together in an image
        
        Args:
            input_image - input RGB image as <np.array>
            label - label binary mask Grayscale image as <np.array>
            prediction - predicted binary mask Grayscale image as <np.array>
            margin - margin between images in terms of pixels in <int>
            save_path - path to save the file <str>
            
        Returns:
            -
            
        Exception:
            -
        '''
        rounded_pred = np.round(prediction)
        margin = np.ones((label.shape[0], margin, 3))
        composite = np.hstack((input_image, margin, self.gray2color(label), margin, self.gray2color(rounded_pred)))
        img = Image.fromarray((composite*255).astype(np.uint8))
        if save_path: save_image()
        return img