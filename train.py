import os
import numpy as np
import random
from PIL import Image
from types import SimpleNamespace

import cifar10_utils

import pytorch_lightning as pl
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

def convertToOneHotNParray(classPredictions_torch):
    # Function: converts a torch class prediction tensor to a one hot coded numpy array
    #   S = batch size
    #   C = number of classes
    #
    # input: pytorch (SxC) tensor representing (unnormalized) class predictions
    # returns: numpy (SxC) one hot coded matrix
    preds=classPredictions_torch.squeeze(dim=1) # get rid of usused dimension
    preds_vec=torch.max(preds,1)[1].cpu().numpy() # extract indices of max values per row (yielding vector of class numbers)
    preds_1hot = np.zeros((preds_vec.size,10)) # set up the one hot encoded numpy array
    preds_1hot[np.arange(preds_vec.size),preds_vec]=1 # use indexing to set the prediced class index to 1
    return  preds_1hot


def accuracy(predictions, targets):
    pred_idx = predictions.argmax(axis=1)   # list, where element i contains column index of maximum element in row i of the predictions matrix
    targ_idx = np.where(targets>0)[1]       # list, where element i contains column of ground truth category of datapoint i
    n_correct = np.sum(pred_idx==targ_idx)
    N = len(pred_idx)
    accuracy = n_correct / N
    return accuracy

class MLP(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        super().__init__()
        self.moduleList=nn.ModuleList()
        n_in=n_inputs
        for n in n_hidden:
          self.moduleList.append(nn.Linear(n_in,n))
          n_in=n
          self.moduleList.append(nn.ELU())
        self.moduleList.append(nn.Linear(n_in,n_classes))   
    def forward(self, x):
        for mod in self.moduleList:
          x=mod(x)
        return x

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # added seed for GPU
        #torch.cuda.manual_seed(42)
        #torch.cuda.manual_seed_all(42)
    else:
        device = torch.device('cpu')
    print('device used:',device)
    pl.seed_everything(42)

    dataset = cifar10_utils.get_cifar10("data/cifar-10-batches-py")
    
    NN=MLP(3*32*32,[100],10)
    loss_module = nn.CrossEntropyLoss()
    
    NN.to(device)
    loss_module.to(device)
    optimizer = torch.optim.Adam(NN.parameters(), lr=1e-3, weight_decay=0.1)

    # Load test data
    x_test,t_test = dataset['test'].next_batch(100)
    x_test=torch.from_numpy(x_test.reshape(x_test.shape[0],-1)).to(device) # flatten and convert test data to torch tensor for processing
    t_test_TorchCP = torch.max(torch.from_numpy(t_test),1)[1].to(device) # format class prediction vector

    for step in range(101):
        x,t = dataset['train'].next_batch(100)
        x=torch.from_numpy(x.reshape(x.shape[0],-1)).to(device)
        t=torch.from_numpy(t).to(device)

        predictions=NN(x)
        loss=loss_module(predictions,torch.max(t,1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation during training (using pre-set frequency)
        if step % 10 == 0:
            # Calculate loss on test data
            with torch.no_grad():
                Y_test=NN(x_test) # forward pass (no grads)
                testLoss = loss_module(Y_test,t_test_TorchCP) # Calculuate loss of predictions Y_test to ground truth class labels
            # Calculate accuracy on test data
            acc=accuracy(convertToOneHotNParray(Y_test),t_test) # Y_test needs to be converted for processing
            # Store for analysis and plotting
            print('\nNumber of SGD steps performed:',step,'/ Training Loss ',loss,'/ Test loss:',testLoss,'/ Test accuracy',acc)
    return

if __name__ == '__main__':
    train()
