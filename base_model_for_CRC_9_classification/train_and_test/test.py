#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 03:17:21 2020

@author: root
"""

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

#import torch.backends.cudnn as cudnn
#import torch.nn.functional as F
import torchvision.models as models
from densenet_ibn_b import densenet121_ibn_b
#import time

import os
#import random
import pickle
import numpy as np
from Focal_Loss import focal_loss
from data_augmention_loader import augmention_dataset
from utils_script import train_predict
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 9, 9
from sklearn.metrics import confusion_matrix,classification_report

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # normalize = transforms.Normalize(mean=[0.736, 0.58, 0.701],std=[0.126, 0.144, 0.113])
    val_trans  = transforms.Compose([transforms.ToTensor()])
    with open('../ETL/fold1_train_list.db','rb') as FP:
        fold1_list = pickle.load(FP)
    with open('../ETL/fold2_train_list.db','rb') as FP:
        fold2_list = pickle.load(FP)
    with open('../ETL/fold3_train_list.db','rb') as FP:
        fold3_list = pickle.load(FP)
    val_list = fold3_list
    train_list = fold1_list + fold2_list
    fold_mark =  'fold_3'
    # val_list = fold2_list
    # train_list = fold1_list + fold2_list
    # fold_mark =  'fold_2'    
    # val_list = fold1_list
    # train_list = fold1_list + fold2_list
    # fold_mark =  'fold_1'   

    test_dset = augmention_dataset(sub_dir = '../DATA/NCT-CRC-HE-100K-NONORM',
                                   class_to_idx = None, 
                                   image_list = train_list,
                                   transform=val_trans)
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size = batch_size, shuffle=False,
        num_workers = 0, pin_memory=False)    
    class_to_label = list(test_dset.class_to_idx)    
    test_dset.shuffle_data(True)
    
    model_mark = 'IBNb'
    if model_mark == 'IBNb':
        model = densenet121_ibn_b(num_classes=len(class_to_label),pretrained = False)
    else:
        model = models.densenet121(num_classes=len(class_to_label),pretrained = False)
    model_dict = torch.load(f"output/{fold_mark}_{model_mark}_densenet_121_CNN_checkpoint_best.pth")

    model = nn.DataParallel(model.cuda())
    model.load_state_dict(model_dict['state_dict'])
    
    criterion = focal_loss(alpha=[1,1,1,1,1,1,1,1,1.0], gamma=2, num_classes = len(class_to_label))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    predict_result,label,loss = train_predict(0, 
                                              test_loader, 
                                              model, 
                                              criterion,
                                              optimizer,
                                              None,
                                              batch_size,
                                              'val',
                                              len(class_to_label))
    
    cr = classification_report(label,np.argmax(predict_result,axis=1),target_names = class_to_label, digits = len(class_to_label))
    print(fold_mark)
    print(cr, "\n")
    
    cm = confusion_matrix(label,np.argmax(predict_result,axis=1))
    plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
    # plt.title('new_ibnb_100K-COLORNORM. Confusion matrix', size=15)
    plt.title(model_mark + ' densenet121 train_split train dataset. Confusion matrix', size=15)
    plt.colorbar()
    tick_marks = np.arange(len(class_to_label))
    plt.xticks(tick_marks, class_to_label,rotation=45, size=10)
    plt.yticks(tick_marks, class_to_label,size=10)
    plt.tight_layout()
    plt.ylabel('Actual label',size=15)
    plt.xlabel('Predicted label',size=15)
    width, height=cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]),xy=(y,x),horizontalalignment='center',verticalalignment='center')
    plt.show()
    
    