import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import imageio
from PIL import Image

import torch
import torch.nn as nn
from torch import optim

from sklearn.model_selection import train_test_split

import help
from Base3DModel import Base3DModel
from MyDataSet import MyDataset
X_train, X_val, y_train, y_val = train_test_split(help.features,help.labels['label'].values,test_size = 0.2,random_state = 42)
train_datasets = MyDataset(datas=X_train,labels=y_train,shape=3,input_D=help.input_D,input_H=help.input_H,input_W=help.input_W,phase='train')
val_datasets = MyDataset(datas=X_val,labels=y_val,shape=3,input_D=help.input_D,input_H=help.input_H,input_W=help.input_W,phase='train')

train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=4, shuffle=False)
print("Data load success")
basemodel_3d = Base3DModel(num_seg_classes=help.num_seg_classes,f=help.basemodel_3d_f)
basemodel_3d
epochs = 100
optimizer = optim.Adam(basemodel_3d.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
help.train_data(basemodel_3d,train_loader,val_loader,epochs,optimizer,scheduler,help.criterion,help.basemodel_3d_checkpoint_path,help.device)

loadmodel = help.load_checkpoint(help.basemodel_3d_checkpoint_path,'basemodel_3d',help.device)

test_datasets = MyDataset(datas=help.temp_data,shape=3,input_D=help.input_D,input_H=help.input_H,input_W=help.input_W,phase='test')

test_loader = torch.utils.data.DataLoader(dataset=test_datasets)
help.all_predict(test_loader,loadmodel,help.device,help.result_3d_basemodel)
