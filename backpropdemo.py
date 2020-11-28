#import os
import numpy as np
#import random
#from PIL import Image
#from types import SimpleNamespace


import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import cifar10_utils
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Callbacks
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class cdataset():
    def __init__(self,type='train'):
        self.dataset = cifar10_utils.get_cifar10(data_dir="data/cifar-10-batches-py",one_hot=False,validation_size=5000)[type]
        self.size=self.dataset.num_examples

    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        return torch.tensor(self.dataset.images[idx]),torch.tensor(self.dataset.labels[idx]).to(torch.long)

class LitMLP(pl.LightningModule):
    def __init__(self,n_inputs, n_hidden, n_classes):
        super().__init__()
        self._create_network(n_inputs,n_hidden,n_classes)
        self._init_params()
    def _create_network(self,in_dim, hiddenLayerList, cls_dim):
        self.moduleList=nn.ModuleList()
        n_in=in_dim
        for n in hiddenLayerList:
          self.moduleList.append(nn.Linear(n_in,n))
          n_in=n
          self.moduleList.append(nn.ELU())
        self.moduleList.append(nn.Linear(n_in,cls_dim))   
    def _init_params(self):
        for m in self.moduleList:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        #x=x.reshape(x.size(0),-1)
        for mod in self.moduleList:
          x=mod(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, t = batch
        x = x.view(x.size(0), -1)
        y = self(x)
        loss = F.cross_entropy(y,t)
        acc=(y.argmax(dim=-1)==t).float().mean()
        self.log('train_acc',acc,on_step=False,on_epoch=True)
        self.log('train_loss', loss)
        return {'loss':loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        train_dataset=cdataset('train')
        trainloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,num_workers=4,shuffle=True)
        return trainloader

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, t = batch
        x = x.view(x.size(0), -1)
        y = self(x)
        loss = F.cross_entropy(y,t)
        #self.log('train_loss', loss)
        return {'val_loss':loss}

    def val_dataloader(self):
        val_dataset=cdataset('validation')
        val_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=32,num_workers=4,shuffle=False)
        return val_loader

    def validation_epoch_end(self,outputs):
        avg_loss=torch.stack([x['val_loss'] for x in outputs]).mean()
        #tensorboard.logs = {'val_loss':avg_loss}
        self.log('val_loss',avg_loss)

input_size=32*32*3
hidden_layers=[100,50,50]
num_classes=10
learning_rate = 1e-3
num_epochs = 5
batch_size=16

if __name__=='__main__':
    #trainer = pl.Trainer(fast_dev_run=True)
    trainer = pl.Trainer(deterministic=True,auto_lr_find=True,max_epochs=num_epochs,fast_dev_run=False)
    model = LitMLP(3*32*32,[100],10)
    trainer.fit(model)
