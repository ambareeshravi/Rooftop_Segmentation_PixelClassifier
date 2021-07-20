'''
Author: Ambareesh Ravi
Date: July 18, 2021
Title: utils.py
Project: For InvisionAI recuritment
Description:
    Contains utility and helper functions for the project
'''

# imports
import numpy as np
import os
from tqdm import tqdm
from time import time
from glob import glob
from PIL import Image

# Global variables
MANUAL_SEED = 42
np.random.seed(42)

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
    return sorted(glob(os.path.join(path, "*")))
    
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

def save_image(array, path, extension = ".png"):
    '''
    Saves an array into an image file
    
    Args:
        array - image as a <np.array>
        path - path for the image as <str>
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
    Image.fromarray(array).save(path)
    
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

def _init_fn(worker_id):
    '''
    Sets the random seed for pytorch dataloader workers
    '''
    np.random.seed(int(MANUAL_SEED))