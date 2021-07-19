'''
Author: Ambareesh Ravi
Date: July 18, 2021
Title: utils.py
Project: For InvisionAI recuritment
Description:
    Contains utility and helper functions for the project
'''

import numpy as np
import os

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