# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 01:13:04 2023

@author: ballahgh
"""

# -*- coding: utf-8 -*-
gpu_mode = 0
n_neural_channels = 192
wt_feature_num = 8
time_duration = '30ms'
day_label = '20190125'    

data_dir_results = 'C:/Models/'
data_dir_data = 'C:/BBData'
data_dir_save = './'    
    
print('*********************Importing Libraries: *********************')
import sys
import os
import numpy as np
from numpy import concatenate
from numpy import *
import h5py
import scipy.io as sio
from scipy.stats import pearsonr
from scipy.io import savemat
import random
import matplotlib
from matplotlib import pyplot
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import init
from torch.autograd import Variable


if gpu_mode == 0:
    device = torch.device('cpu')
if gpu_mode == 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

################################################################# General Functions ##########################
def target_normalization_centralization(target):
    # normalize and center the targets
    target = target - target.min(axis=0)
    for j in range(target.shape[1]):
        if target[:,j].max(axis=0) != 0:
            target[:,j] = 2*(target[:,j]/target[:,j].max(axis=0) - 0.5)
    return target

#############################################################################
class LoadModel_exp2(nn.Module):
    def __init__(self, n_neural_channels, wt_feature_num):
        super(LoadModel_exp2, self).__init__()
        self.hpf1  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2 , bias = False)
        self.lpf1  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2, bias = False)
        self.hpf2  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2 , bias = False)
        self.lpf2  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2, bias = False)    
        self.hpf3  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2 , bias = False)
        self.lpf3  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2, bias = False)
        self.hpf4  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2 , bias = False)
        self.lpf4  = nn.Conv1d(1, 1, kernel_size= 40, stride = 2, bias = False) 
        self.hpf5 = nn.Conv1d(1, 1, kernel_size= 40, stride = 2 , bias = False)
        self.lpf5 = nn.Conv1d(1, 1, kernel_size= 40, stride = 2, bias = False) 
        self.hpf6 = nn.Conv1d(1, 1, kernel_size= 40, stride = 2 , bias = False)
        self.lpf6 = nn.Conv1d(1, 1, kernel_size= 40, stride = 2, bias = False) 
        self.hpf7 = nn.Conv1d(1, 1, kernel_size= 40, stride = 2 , bias = False)
        self.lpf7 = nn.Conv1d(1, 1, kernel_size= 40, stride = 2, bias = False) 
        self.pool = nn.MaxPool1d(10,stride = 2)

        self.pad = nn.ConstantPad1d((38, 39), 0)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(n_neural_channels*wt_feature_num, affine = False, track_running_stats = False)
        self.nl = nn.LeakyReLU(-1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        
        #1
        x = x.view(-1, 1, x.shape[2])
        x = self.pad(x)
        x_h = self.hpf1(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf1(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x_feat = self.adaptive_avg_pool(self.nl(x_h))
        
        #2
        x = self.pad(x)
        x_h = self.hpf2(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf2(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #3
        x = self.pad(x)
        x_h = self.hpf3(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf3(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #4
        x = self.pad(x)
        x_h = self.hpf4(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf4(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #5
        x = self.pad(x)
        x_h = self.hpf5(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf5(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #6
        x = self.pad(x)
        x_h = self.hpf6(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf6(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #7
        x = self.pad(x)
        x_h = self.hpf7(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf7(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 

        x = x.view(batch_size, in_channels, x.shape[2])
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x))), dim = 2)   
        x_feat = x_feat.view(-1,in_channels*x_feat.shape[2])

        x_feat = self.bn(x_feat)
        return x_feat, self.lpf1.weight.data, self.hpf1.weight.data, self.lpf2.weight.data, self.hpf2.weight.data, self.lpf3.weight.data, self.hpf3.weight.data, self.lpf4.weight.data, self.hpf4.weight.data, self.lpf5.weight.data, self.hpf5.weight.data, self.lpf6.weight.data, self.hpf6.weight.data, self.lpf7.weight.data, self.hpf7.weight.data       
        
class NewModel_exp2(nn.Module):
    def __init__(self, old_model):
        super(NewModel_exp2,self).__init__()
        child_counter = 0
        for child in old_model.children():
            if child_counter == 0:
                self.hpf1 = child
            elif child_counter == 1:
                self.lpf1 = child
            elif child_counter == 2:
                self.hpf2 = child
            elif child_counter == 3:
                self.lpf2 = child                
            elif child_counter == 4:
                self.hpf3 = child
            elif child_counter == 5:
                self.lpf3 = child                   
            elif child_counter == 6:
                self.hpf4 = child
            elif child_counter == 7:
                self.lpf4 = child                  
            elif child_counter == 8:
                self.hpf5 = child
            elif child_counter == 9:
                self.lpf5 = child  
            elif child_counter == 10:
                self.hpf6 = child
            elif child_counter == 11:
                self.lpf6 = child   
            elif child_counter == 12:
                self.hpf7 = child
            elif child_counter == 13:
                self.lpf7 = child                  
            elif child_counter == 14:
                self.pool = child
            elif child_counter == 15:
                self.pad = child
            elif child_counter == 16:
                self.adaptive_avg_pool = child
            elif child_counter == 17:
                self.bn = child
            elif child_counter == 18:
                self.nl = child    
            elif child_counter == 19:
                break
            child_counter = child_counter + 1
    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        
        #1
        x = x.view(-1, 1, x.shape[2])
        x = self.pad(x)
        x_h = self.hpf1(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf1(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x_feat = self.adaptive_avg_pool(self.nl(x_h))
        
        #2
        x = self.pad(x)
        x_h = self.hpf2(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf2(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #3
        x = self.pad(x)
        x_h = self.hpf3(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf3(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #4
        x = self.pad(x)
        x_h = self.hpf4(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf4(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #5
        x = self.pad(x)
        x_h = self.hpf5(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf5(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #6
        x = self.pad(x)
        x_h = self.hpf6(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf6(x)
        x = F.dropout(x, p = 0.2, training = self.training)        
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 
        
        #7
        x = self.pad(x)
        x_h = self.hpf7(x)
        x_h = x_h.view(batch_size, in_channels, x_h.shape[2])
        x = self.lpf7(x)
        x = F.dropout(x, p = 0.2, training = self.training)
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x_h))), dim = 2) 

        x = x.view(batch_size, in_channels, x.shape[2])
        x_feat = torch.cat((x_feat, self.adaptive_avg_pool(self.nl(x))), dim = 2)   
        x_feat = x_feat.view(-1,in_channels*x_feat.shape[2])

        x_feat = self.bn(x_feat)
        return x_feat, self.lpf1.weight.data, self.hpf1.weight.data, self.lpf2.weight.data, self.hpf2.weight.data, self.lpf3.weight.data, self.hpf3.weight.data, self.lpf4.weight.data, self.hpf4.weight.data, self.lpf5.weight.data, self.hpf5.weight.data, self.lpf6.weight.data, self.hpf6.weight.data, self.lpf7.weight.data, self.hpf7.weight.data 
    

def Loss_Function(X, y):
    criterion = nn.MSELoss(reduction='mean')
    X_numpy = X
    y_numpy = y
    X_numpy = X_numpy.cpu().detach().numpy()
    y_numpy = y_numpy.cpu().detach().numpy()
    reg = LinearRegression().fit(X_numpy, y_numpy)
    others_beta_x = reg.coef_[0,:].reshape((reg.coef_.shape[1],1))
    others_beta_y = reg.coef_[1,:].reshape((reg.coef_.shape[1],1))
    b0_x = reg.intercept_[0]
    b0_y = reg.intercept_[1]
    if gpu_mode == 1:
        others_beta_x = Variable(torch.Tensor(others_beta_x).type(torch.FloatTensor), requires_grad=False).cuda()
        others_beta_y = Variable(torch.Tensor(others_beta_y).type(torch.FloatTensor), requires_grad=False).cuda()
        b0_x = Variable(torch.Tensor([b0_x]).type(torch.FloatTensor), requires_grad=False).cuda()
        b0_y = Variable(torch.Tensor([b0_y]).type(torch.FloatTensor), requires_grad=False).cuda()    
    else:
        others_beta_x = Variable(torch.Tensor(others_beta_x).type(torch.FloatTensor), requires_grad=False)
        others_beta_y = Variable(torch.Tensor(others_beta_y).type(torch.FloatTensor), requires_grad=False)
        b0_x = Variable(torch.Tensor([b0_x]).type(torch.FloatTensor), requires_grad=False)
        b0_y = Variable(torch.Tensor([b0_y]).type(torch.FloatTensor), requires_grad=False)       

    pred_x = torch.matmul(X, others_beta_x) + b0_x
    pred_y = torch.matmul(X, others_beta_y) + b0_y
    pred = torch.cat((pred_x, pred_y), axis = 1)
    loss = criterion(pred, y)    
    return loss, pred, others_beta_x, others_beta_y, b0_x, b0_y

def Loss_Function_Test(X, y, others_beta_x, others_beta_y, b0_x, b0_y):
    criterion = nn.MSELoss(reduction='mean')
    pred_x = torch.matmul(X, others_beta_x) + b0_x
    pred_y = torch.matmul(X, others_beta_y) + b0_y
    pred = torch.cat((pred_x, pred_y), axis = 1)
    loss = criterion(pred, y) 
    return loss, pred
 
    
old_model = LoadModel_exp2(n_neural_channels, wt_feature_num)
old_model.load_state_dict(torch.load(data_dir_results + 'Latest_FENet_Model.pth', map_location = device))
cnn_model = NewModel_exp2(old_model)

