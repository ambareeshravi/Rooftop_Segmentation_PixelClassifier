'''
Author: Ambareesh Ravi
Date: July 18, 2021
Title: generate_dataset.py
Project: For InvisionAI recuritment
Description:
    Generates a dataset of image patches from the given image and label
'''

# library imports
from PIL import Image
from tqdm import tqdm

# module imports
from utils import *

def generate_patches(data, label, patch_size = 512, patch_stride = 64, path = "./dataset/"):
    '''
    Generates and save patches of images from a parent image pair of data and label
    
    Args:
        data - <np.array> containing the original input image
        label - <np.array> containing the binary mask for the input
        patch_size - size of the square patch to be generated as <int>
        patch_stride - stride for moving across the parent image as <int>
        path - directory to save the patches as <str>
    
    Returns:
        -
    
    Exception:
        number of patches as <int>
    '''
    
    # Setup paths
    create_directory(path)
    data_path = os.path.join(path, "data")
    create_directory(data_path)
    label_path = os.path.join(path, "label")
    create_directory(label_path)
    
    # Loop along the dimensions of the image
    patch_count = 0
    for xp in tqdm(range(0, data.shape[0], patch_stride)):
        for yp in range(0, data.shape[1], patch_stride):
            
            # Crop the data and label patches
            data_patch = data[xp:(xp + patch_size), yp:(yp + patch_size)]
            label_patch = label[xp:(xp + patch_size), yp:(yp + patch_size)]
            
            # Check if there is at least one roof from the binary mask and for the output shape
            if np.sum(label_patch) > 0 and (data_patch.shape == (patch_size, patch_size, data.shape[-1])):
                file_name = "%05d.png"%(patch_count)
                # Save the files to the desired locations
                save_image(data_patch, os.path.join(data_path, file_name))
                save_image(label_patch, os.path.join(label_path, file_name))
                patch_count += 1
    
    # return the patch count
    return patch_count

if __name__ == '__main__':
    data = np.array(Image.open("image.tif").convert("RGB"))
    label = np.array(Image.open("labels.tif"))
    generate_patches(data, label)