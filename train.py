'''
Author: Ambareesh Ravi
Date: July 19, 2021
Title: train.py
Project: For InvisionAI recuritment
Description:
    Contains the class to train the pytorch PixelClassifier model
'''

import torch
from torch import nn

from utils import *
from models import *
from data import *

def get_loss(loss_type, reduction):
    '''
    Returns <torch.nn> loss function
    
    Possible loss functions apart from BCE:
        1) MSE 2) SSIM? 3) Focal Loss? 4) DICE loss?

    Args:
        loss_type - type of loss function as <str>
        reduction - type of reduction [mean/ sum] as <str>

    Returns:
        loss function

    Exception:
        -
    '''
    loss_type = loss_type.lower()

    if "bce" in loss_type:
        return nn.BCELoss(reduction = reduction)
    elif "mse" in loss_type:
        return nn.MSELoss(reduction = reduction)
        
class PixelClassifier_Trainer:
    def __init__(
        self,
        model = None,
        model_path = "",
        train_loader = None,
        val_loader = None,
        optimizer = "adam",
        loss_params = {
         "loss_type": "bce",
         "reduction": "sum"
        },
        useGPU = True,
        showStatus = False
    ):
        '''
        Initializes the class that trains the pixel classifier CNN model
        
        Args:
            model - <torch.nn.Module> model
            model_path - path to the model <str>
            
        Returns:
            -
        
        Exception:
            -
        '''
        self.model = PixelClassifier_CNN() if model == None else model
        self.model_path = model_path
        
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            INFO("Running training on GPU")
        self.model.to(self.device)
        
        self.showStatus = showStatus
        
        if self.showStatus: INFO("Model ready")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.loss_params = loss_params
        self.optimizer = optimizer
        
        self.history = {
            "train_loss": list(),
            "validation_loss": list()
        }
        
    def get_optimizer(self, lr):
        '''
        Creates and returns an optimizer for training
        
        Args:
            lr - learning rate as <float>
            
        Returns:
            -
        
        Exception:
            -
        '''
        if "sgd" in self.optimizer.lower():
            return torch.optim.SGD(self.model.parameters(), lr = lr, momentum=0.9)
        else: # default is Adam
            return torch.optim.Adam(self.model.parameters(), lr = lr)
    
    def step(self, inputs, labels):
        '''
        Step for training or prediction
        
        Args:
            inputs - inputs as <torch.Tensor>
            labels - labels as <torch.Tensor>
            
        Returns:
            loss as <torch.Tensor>
        
        Exception:
            -
        '''
        predictions = self.model(inputs)
        return self.loss_criterion(predictions, labels)
    
    def train_step(self, images, labels):
        '''
        Train step with forward pass and gradient updation
        
        Args:
            inputs - inputs as <torch.Tensor>
            labels - labels as <torch.Tensor>
            
        Returns:
            loss as <torch.Tensor>
        
        Exception:
            -
        '''
        self.model.train()
        self.model.zero_grad()
        loss = self.step(images.to(self.device), labels.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def val_step(self, images, labels):
        '''
        Validation step with forward pass with no gradient updation
        
        Args:
            inputs - inputs as <torch.Tensor>
            labels - labels as <torch.Tensor>
            
        Returns:
            loss as <torch.Tensor>
        
        Exception:
            -
        '''
        self.model.eval()
        with torch.no_grad():
            loss = self.step(images.to(self.device), labels.to(self.device))
        return loss.item()
    
    def display_epoch_stat(self, epoch, train_loss, val_loss, epoch_time):
        '''
        Displays the stats
        
        Args:
            epoch - current epoch <int>
            train_loss - training loss as <float>
            val_loss - validation loss as <float>
            epoch_time - time in seconds <int>
            
        Returns:
            -
        
        Exception:
            -
        '''
        print("Epoch: [%d/%d] | Train Loss: %0.4f | Val Loss: %0.4f | Time Elapsed(s): %0.2f"%(epoch, self.epochs, train_loss, val_loss, epoch_time))
        print("-"*40)
    
    def train(self, lr = 1e-4, epochs = 100, status_frequency = 5):
        '''
        Function to train the model
        
        Args:
            lr - learning rate in <float>
            epochs - Number of epochs as <int>
            status_frequency - frequency at which the status has to be displayed <int>
            
        Returns:
            -
        
        Exception:
            -
        '''
        self.epochs = epochs
        
        # Get loss and optimizer ready
        self.loss_criterion = get_loss(self.loss_params["loss_type"], self.loss_params["reduction"])
        self.optimizer = self.get_optimizer(lr)
        
        if self.showStatus: INFO("Loss and Optimizer ready")
        if self.showStatus: INFO("Starting the training")
        
        # Start training
        for epoch in range(1, epochs + 1):
            epoch_st = time()
            
            epoch_train_loss, epoch_val_loss = list(), list()
            
            # Training step
            for train_batch_idx, (train_images, train_labels) in tqdm(enumerate(self.train_loader)):
                train_loss = self.train_step(train_images, train_labels)
                epoch_train_loss.append(train_loss)
            
            # Validation step
            for val_batch_idx, (val_images, val_labels) in tqdm(enumerate(self.val_loader)):
                val_loss = self.val_step(val_images, val_labels)
                epoch_val_loss.append(val_loss)
            
            epoch_et = time()
            
            # Calulate the mean loss for epoch
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_val_loss = np.mean(epoch_val_loss)
            
            # Record history
            self.history["train_loss"].append(epoch_train_loss)
            self.history["validation_loss"].append(epoch_val_loss)
            
            # Display stats
            if epoch > 1 or epoch%status_frequency == 0:
                self.display_epoch_stat(epoch, epoch_train_loss, epoch_val_loss, (epoch_et-epoch_st))
                if epoch == 1:
                    INFO("Expected time of completion of train: %0.2f hours"%((epoch_et-epoch_st)*self.epochs//3600))
        
        torch.save(self.model, self.model_path)
        return self.history
    
if __name__ == '__main__':
    # Parse input arguments from the user for traing the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = "models/model.pth", help="Path to store the trained model")
    parser.add_argument("--batch_size", type=int, default = 32, help="Batch Size for training")
    parser.add_argument("--epochs", type=int, default = 100, help="NUmber of epochs for training")
    args = parser.parse_args()
    
    # Get dataset
    train_dataset = Rooftop_Dataset(isTrain = True)
    # Get data loader
    train_loader, val_loader = get_data_loader(train_dataset, batch_size = args.batch_size)
    
    # Create an instance of the trainer
    trainer = PixelClassifier_Trainer(
        model_path=args.model_path,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer='adam',
        loss_params={'loss_type': 'bce', 'reduction': 'sum'},
        useGPU=True,
        showStatus=True,
    )
    
    # Train the model
    history = trainer.train(epochs = args.epochs, status_frequency=1)