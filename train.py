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

class PixelClassifier_Trainer:
    def __init__(
        self,
        model = None,
        model_path = "",
        train_loader = None,
        val_loader = None,
        optimizer = "adam",
        loss_params = {
         "bce_reduction": "mean"
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
        self.model.to(self.device)
        
        self.showStatus = showStatus
        
        if self.showStatus: print("Model ready")
        
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
        description
        
        Args:
            -
            
        Returns:
            -
        
        Exception:
            -
        '''
        if "sgd" in self.optimizer.lower():
            return torch.optim.SGD(self.model.parameters(), lr = lr, momentum=0.9)
        else:
            return torch.optim.Adam(self.model.parameters(), lr = lr)
    
    def get_loss(self,):
        '''
        description
        
        Args:
            -
            
        Returns:
            -
        
        Exception:
            -
        '''
        return nn.BCEWithLogitsLoss(reduction = self.loss_params["bce_reduction"])
    
    def step(self, inputs, labels):
        '''
        description
        
        Args:
            -
            
        Returns:
            -
        
        Exception:
            -
        '''
        predictions = self.model(inputs)
        return self.loss_criterion(predictions, labels)
    
    def train_step(self, images, labels):
        '''
        description
        
        Args:
            -
            
        Returns:
            -
        
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
        description
        
        Args:
            -
            
        Returns:
            -
        
        Exception:
            -
        '''
        self.model.eval()
        with torch.no_grad():
            loss = self.step(images.to(self.device), labels.to(self.device))
        return loss.item()
    
    def display_epoch_stat(self, epoch, train_loss, val_loss, epoch_time):
        '''
        description
        
        Args:
            -
            
        Returns:
            -
        
        Exception:
            -
        '''
        print("Epoch: [%d/%d] | Train Loss: %0.4f | Val Loss: %0.4f | Time Elapsed(s): %0.2f"%(epoch, self.epochs, train_loss, val_loss, epoch_time))
    
    def train(self, lr = 1e-4, epochs = 100, status_frequency = 5):
        '''
        description
        
        Args:
            -
            
        Returns:
            -
        
        Exception:
            -
        '''
        self.epochs = epochs
        
        self.loss_criterion = self.get_loss()
        self.optimizer = self.get_optimizer(lr)
        
        if self.showStatus: print("Loss and Optimizer ready")
        
        if self.showStatus: print("Starting the training")
            
        for epoch in range(1, epochs + 1):
            epoch_st = time()
            
            epoch_train_loss, epoch_val_loss = list(), list()
            for train_batch_idx, (train_images, train_labels) in tqdm(enumerate(self.train_loader)):
                train_loss = self.train_step(train_images, train_labels)
                epoch_train_loss.append(train_loss)
            
            for val_batch_idx, (val_images, val_labels) in tqdm(enumerate(self.val_loader)):
                val_loss = self.val_step(val_images, val_labels)
                epoch_val_loss.append(val_loss)
            
            epoch_et = time()
            
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_val_loss = np.mean(epoch_val_loss)
            
            self.history["train_loss"].append(epoch_train_loss)
            self.history["validation_loss"].append(epoch_val_loss)
            
            if epoch > 1 or epoch%status_frequency == 0:
                self.display_epoch_stat(epoch, epoch_train_loss, epoch_val_loss, (epoch_et-epoch_st))
        
        torch.save(self.model, self.model_path)
        return self.history