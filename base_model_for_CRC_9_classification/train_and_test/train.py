#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:38:42 2020

@author: Bohrium.Kwong
"""
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
#import torch.nn.functional as F
import torchvision.models as models
from densenet_ibn_b import densenet121_ibn_b
import time
import copy
import os
#import random
import pickle
import numpy as np

#from ghost_net import ghost_net
from data_augmention_loader import augmention_dataset
from utils_script import calc_accuracy,train_predict
from Focal_Loss import focal_loss
import argparse

def main(parser):
    global args, best_acc
    args = parser.parse_args()
    best_acc = -3
#    device_ids = list(range(torch.cuda.device_count()))
    
    if not os.path.isdir(args.output):os.makedirs(args.output)
    
    train_trans = transforms.Compose([transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                        ])
    val_trans = transforms.Compose([transforms.ToTensor()])

    
    train_dset = augmention_dataset(sub_dir = args.root_dir,class_to_idx = None, image_list = args.train_list,
                                    transform=train_trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size = args.batch_size, shuffle=False,
        num_workers = args.workers, pin_memory=False)    
    class_to_label = list(train_dset.class_to_idx)    
    train_dset.shuffle_data(True)
    train_dset.setmode(2)
    train_dset.maketraindata(5)
    print('Total of train set after data expansion: {}'.format(len(train_dset)))

    val_dset = augmention_dataset(sub_dir = args.root_dir,class_to_idx = None, image_list = args.val_list,
                                    transform=val_trans)
    val_dset.setmode(1)
    val_dset.shuffle_data(True)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size = args.batch_size, shuffle=False,
        num_workers = args.workers, pin_memory=False)     

    # time_mark = args.fold_mark
    if args.model_mark == 'IBNb':
        model = densenet121_ibn_b(num_classes=len(class_to_label),pretrained = False)
    else:
        model = models.densenet121(num_classes=len(class_to_label),pretrained = False)
    model = nn.DataParallel(model.cuda())
    # model = model.cuda()
    # model.load_state_dict(model_dict['state_dict'])
    
    criterion = focal_loss(alpha=[1,1,1,1,1,1,1,1,1.10], gamma=2, num_classes = len(class_to_label))
#    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#    optimizer = nn.DataParallel(optimizer)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma= 0.1, milestones= [92, 136])

    cudnn.benchmark = True
    
    # time_mark = time.strftime('%Y_%m_%d_',time.localtime(time.time()))

    if not os.path.isdir(args.output): os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, args.fold_mark + args.model_mark +  '_densenet_121__focal_loss.csv')): 
        fconv = open(os.path.join(args.output, args.fold_mark + args.model_mark + '_densenet_121__focal_loss.csv'), 'w')
        fconv.write(' ,Training,,,,Validation,,,\n')
        fconv.write('epoch,train_acc,train_recall,train_fnr,train_loss,val_acc,val_recall,val_fnr,val_loss')
        fconv.close()

    early_stop_count = 0
    
    #loop throuh epochs
    for epoch in range(args.nepochs):
        if epoch >=2/3*args.nepochs and early_stop_count >= 4:
            print('Early stop at Epoch:'+ str(epoch+1))
            break
        start_time = time.time()
        if epoch > 2:
            # train_dset.maketraindata(5)
            train_dset.shuffle_data(True)
            
        predict_result,label,loss = train_predict(epoch, 
                                                  train_loader, 
                                                  model, 
                                                  criterion,
                                                  optimizer,
                                                  scheduler,
                                                  args.batch_size,
                                                  'train',
                                                  len(class_to_label))
        
        metrics_meters = calc_accuracy(np.argmax(predict_result,axis=1), label)
        print('\tTraining  Epoch: [{}/{}] Acc: {} Recall:{} Fnr:{} Loss: {}'.format(epoch+1, \
              args.nepochs, metrics_meters['acc'],metrics_meters['recall'],metrics_meters['fnr'],loss))
        result_record = '\n'+ str(epoch+1) + ',' + str(metrics_meters['acc']) + ',' + str(metrics_meters['recall']) + ','\
                + str(metrics_meters['fnr']) + ',' + str(loss)
                
        predict_result,label,loss = train_predict(epoch, 
                                                  val_loader, 
                                                  model, 
                                                  criterion,
                                                  optimizer,
                                                  None,
                                                  args.batch_size,
                                                  'val',
                                                  len(class_to_label))
        
        metrics_meters = calc_accuracy(np.argmax(predict_result,axis=1), label)
        print('\tTesting  Epoch: [{}/{}] Acc: {} Recall:{} Fnr:{} Loss: {}'.format(epoch+1, \
              args.nepochs, metrics_meters['acc'],metrics_meters['recall'],metrics_meters['fnr'],loss))
        
        result_record = result_record + ',' + str(metrics_meters['acc']) + ',' + str(metrics_meters['recall']) + ','\
                + str(metrics_meters['fnr']) + ',' + str(loss)
                
        fconv = open(os.path.join(args.output, args.fold_mark + args.model_mark + '_densenet_121__focal_loss.csv'), 'a')
        fconv.write(result_record)
        fconv.close()
        
        tmp_acc = (metrics_meters['acc'] + metrics_meters['recall'])/2 - loss - metrics_meters['fnr']
        if tmp_acc >= best_acc:
            best_acc = copy.copy(tmp_acc)
            obj = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'class_to_idx' : train_dset.class_to_idx
            }
            torch.save(obj, os.path.join(args.output, args.fold_mark + args.model_mark +'_densenet_121_CNN_checkpoint_best.pth'))
        else:
            if epoch >=2/3*args.nepochs:
                early_stop_count += 1
            
        print('\tEpoch %d has been finished, needed %.2f sec.' % (epoch + 1,time.time() - start_time))


    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # if batch_size >=60,you need more than 12G GPU memory
    parser = argparse.ArgumentParser(description='2022 CRC_9_class classifier training script')
    with open('../ETL/fold1_train_list.db','rb') as FP:
        fold1_list = pickle.load(FP)
    with open('../ETL/fold2_train_list.db','rb') as FP:
        fold2_list = pickle.load(FP)
    with open('../ETL/fold3_train_list.db','rb') as FP:
        fold3_list = pickle.load(FP)
    val_list = fold3_list
    train_list = fold2_list + fold1_list
    parser.add_argument('--train_list', type=list, default=train_list,help ='train data list library binary')
    parser.add_argument('--val_list', type=list, default=val_list,help='val data list library binary')
    parser.add_argument('--fold_mark', type=str, default='fold_3_')
    parser.add_argument('--model_mark', type=str, default='IBNb',
                        help="should be 'IBNb'(means for using IBN structure) or 'ori'(means for original densenet)")
    # base model is DenseNet 121
    parser.add_argument('--output', type=str, default='./output', help='name of output file')
    parser.add_argument('--batch_size', type=int, default=135, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--root_dir', type=str, default='../DATA/NCT-CRC-HE-100K-NONORM', help='root_dir of data')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    main(parser)
