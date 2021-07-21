'''
Author: Ambareesh Ravi
Date: July 18, 2021
Title: generate_dataset.py
Project: For InvisionAI recuritment
Description:
    Generates a dataset of image patches from the given image and label
    Generates patches at multiple scales and different rotations
'''

# library imports
from PIL import Image
from tqdm import tqdm
import shutil

# module imports
from utils import *

# Functions to create dataset from scratch

def generate_patches(data, label, patch_size = 512, stride_fraction = 0.25, resize_to = (512, 512), parent_path = "./rooftop/"):
    '''
    Generates and save patches of images from a parent image pair of data and label at multiple scales and strides
    
    Args:
        data - <np.array> containing the original input image
        label - <np.array> containing the binary mask for the input
        patch_size - size of the square patch to be generated as <int>
        stride_fraction - stride for moving across the parent image as <float> i.e a fraction of patch_size
        resize_to - size to which the patches have to be resized as <tuple> of <int> (w,h)
        parent_path - directory to save the patches as <str>
    
    Returns:
        number of patches as <int>
    
    Exception:
        -
        
    # Add resizing images of multiple scales and saving
    '''
    
    patch_stride = int(stride_fraction * patch_size)
    
    # Setup paths
    create_directory(parent_path)
    
    for type_ in ["train", "test"]:
        create_directory(os.path.join(parent_path, type_))
        for d in ["data", "label"]:
            create_directory(os.path.join(parent_path, type_, d))
            
    data_path = os.path.join(parent_path, "train", "data")
    label_path = os.path.join(parent_path, "train", "label")
    
    # Loop along the dimensions of the image
    patch_count = len(read_directory_content(data_path))
    init_patch_count = patch_count
    for xp in tqdm(range(0, data.shape[0], patch_stride)):
        for yp in range(0, data.shape[1], patch_stride):
            
            # Crop the data and label patches
            data_patch = data[xp:(xp + patch_size), yp:(yp + patch_size)]
            label_patch = label[xp:(xp + patch_size), yp:(yp + patch_size)]
            
            # Check for the output shape
            if (data_patch.shape == (patch_size, patch_size, data.shape[-1])): # and np.sum(label_patch) > 0:
                file_name = "%05d.png"%(patch_count)
                # Save the files to the desired locations
                save_image(data_patch, os.path.join(data_path, file_name), resize = resize_to)
                save_image(label_patch, os.path.join(label_path, file_name), resize = resize_to)
                patch_count += 1
    
    # return the patch count
    return patch_count-init_patch_count

def split_data(data_path = "rooftop/", test_size = 0.1):
    '''
    Splits the generated data randomly into train and test sets based on the given fraction
    
    Args:
        data_path - path to the dataset as <str>
        test_size - fraction of the test set size as <float>
    
    Returns:
        -
    
    Exception:
        Throws an exception if there is a mismatch of files between the data and labels
        
    '''
    # Read files
    files_list = read_directory_content(os.path.join(data_path, "train", "data"))
    INFO("Number of test files: %d"%(int(len(files_list)*test_size)))
    
    # Randomly pick test files to be moved
    test_files = np.random.choice(files_list, size = int(len(files_list)*test_size))
    
    # Move the files
    for tf in test_files:
        # Check the presence of the files
        if not (os.path.isfile(tf) or os.path.isfile(tf.replace("data", "label"))): continue
        
        # Move
        shutil.move(tf, tf.replace("train", "test"))
        shutil.move(tf.replace("data", "label"), tf.replace("data", "label").replace("train", "test"))
    
    # Check if all files available in "data/" are also available in "label/"
    check = [os.listdir(os.path.join(data_path, tp, "data")) == os.listdir(os.path.join(data_path, tp, "label")) for tp in ["train", "test"]]
    assert all(check), "Files mismatch"
    
def create_dataset(soure_data = "source_data/", data_path = "rooftop/", patch_info= [(512, 0.25)], rotations = [0, 90, 180], test_size = 0.15):
    '''
    Creates a dataset from the given two images [data.tif/ label.tif]
    
    Args:
        soure_data - path with the original images of rooftop aerial view and corresponding label as <str>
        data_path - path to the dataset as <str>
        patch_info - contains the configurations for patch generation as a <list> of (patch size <int>, stride fraction as <float>)
        rotations - <list> of image rotations in degrees as <int>
        test_size - fraction of the test set size as <float> to split the dataset
    
    Returns:
        -
    
    Exception:
        -
        
    '''
    # Read the original images
    d_img = Image.open(os.path.join(soure_data, "image.tif")).convert("RGB")
    l_img = Image.open(os.path.join(soure_data, "labels.tif"))
    
    # Apply different rotations
    for rotation in rotations:
        data = np.array(d_img.rotate(rotation))
        label = np.array(l_img.rotate(rotation))
        
        # Apply different patch sizes
        for (patch_size, stride_fraction) in patch_info:
            
            # Generate patches and save them
            n_patches = generate_patches(data, label, patch_size = patch_size, stride_fraction = stride_fraction, parent_path = data_path)
            INFO("Generated %d patches of size %d, stride %d and rotation %d"%(n_patches, patch_size, int(stride_fraction*patch_size), rotation))
    
    # Spit the created dataset
    split_data(data_path, test_size = test_size)
        
if __name__ == '__main__':
    data_path = "rooftop/"
    
    patch_info = [(512,0.25), (1024,0.25), (256,0.5)]
    rotations = [0, 90, 180]
    
    create_dataset(data_path, patch_info, rotations)