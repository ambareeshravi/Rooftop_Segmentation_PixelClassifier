'''
Author: Ambareesh Ravi
Date: July 20, 2021
Title: data.py
Description:
    Contains data handling functionalities for training and evaluation of models
'''

# Library imports
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

# Module imports
from utils import *
from generate_dataset import *

class Rooftop_Dataset(Dataset):
    def __init__(self, data_path = "rooftop/", isTrain = True, image_size = (512, 512), useNormalized = False):
        '''
        Rooftop Dataset  - creates torch.utils.data.Dataset
        
        Args:
            data_path - path to the dataset as <str>
            isTrain - <bool> to indicate train/test modes and loads the data accordingly
            image_size - input image resolution as <tuple>
            useNormalized - <bool> to normalize the tensor
            
        Returns:
            -
            
        Exception:
            -
        '''
        super().__init__()
        self.data_path = os.path.join(data_path, "train" if isTrain else "test")
        self.image_size = image_size
        
        # rotation, angle, position, mirror, flipping
        transforms_list = [
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()
        ]
        
        if useNormalized: transforms_list += [transforms.Normalize([0.5], [0.5])]
        self.data_transforms = transforms.Compose(transforms_list)
        
        # If dataset does not exist, create one
        if not os.path.exists(os.path.join(self.data_path, "data")):
            create_dataset()
            
        self.data = read_directory_content(os.path.join(self.data_path, "data"))
        self.labels = read_directory_content(os.path.join(self.data_path, "label"))
    
    def transform_image(self, image):
        '''
        Applies data transform to the PIL Image
        
        Args:
            image - input <PIL.Image>
            
        Returns:
            transformed image as <torch.Tensor>
        
        Exception:
            -
        '''
        return self.data_transforms(image)
    
    def get_image(self, file_path):
        '''
        Reads the image and returns the transformed image
        
        Args:
            file_path - path to the image as <str>
            
        Returns:
            transformed image as <torch.Tensor>
        
        Exception:
            -
        '''
        return self.transform_image(read_image(file_path)) 
        
    def __len__(self):
        '''
        Returns the length of the dataset
        
        Args:
            -
            
        Returns:
            length as <int>
        
        Exception:
            -
        '''
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        returns the item (data, label) at an index
        
        Args:
            idx - index as <int>
            
        Returns:
            (data, label)
        
        Exception:
            -
        '''
        return self.get_image(self.data[idx]), self.get_image(self.labels[idx])

# Helper functions
def _init_fn(worker_id):
    '''
    Sets the random seed for pytorch dataloader workers
    '''
    np.random.seed(int(MANUAL_SEED))
    
def get_data_loader(
    data,
    batch_size = 32,
    val_split = 0.1,
    num_workers = 4,
    isTrain = True
):
    '''
    Applies dataset to dataloader functionality

    Args:
        data - the dataset in <torch.utils.data.Dataset> format
        batch_size - size of the data batches as <int>
        val_split - [optional] fraction of split between train/val sets as <float>
        num_workers - Number of background workers for the data loader as <int>
        isTrain - <bool> to indicate train/test datasets

    Returns:
        train and val loaders if isTrain is True
        test loader if isTrain is False

    Exception:
        -
    '''
    if not isTrain:
        return DataLoader(data, batch_size = batch_size, shuffle=False, num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
        
    split_point = int((1-val_split)*len(data))
    train_data, val_data = random_split(data, [split_point, len(data)-split_point])
    
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=False, num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
    
    return train_loader, val_loader
