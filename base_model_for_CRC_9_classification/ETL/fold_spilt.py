# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:20:32 2020

@author: kwong
"""

import os
import glob
import pickle
from tqdm import tqdm
import random

def get_list(img_dir):                   
    img_list = glob.glob(os.path.join(img_dir, "*.tif"))
    pbar = tqdm(total = len(img_list))
    fold1_list ,fold2_list,fold3_list = [],[],[]
    for file_path in img_list:
        random_flag = random.randint(1,999)
        if random_flag % 3 == 1:
            fold1_list.append(file_path)
        elif random_flag % 3 == 2:
            fold2_list.append(file_path)
        else:
            fold3_list.append(file_path)
        pbar.update(1)
    pbar.close()        
    return fold1_list,fold2_list, fold3_list

def get_list_base_on_train_list(train_list):
    pbar = tqdm(total = len(train_list))
    fold1_list ,fold2_list,fold3_list = [],[],[]
    for file_path in train_list:
        random_flag = random.randint(1,999)
        if random_flag % 3 == 1:
            fold1_list.append(file_path)
        elif random_flag % 3 == 2:
            fold2_list.append(file_path)
        else:
            fold3_list.append(file_path)
        pbar.update(1)
    pbar.close()        
    return fold1_list,fold2_list, fold3_list    

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # plt.rcParams['figure.figsize'] = 14, 14
    sub_dir = '../DATA/NCT-CRC-HE-100K-NONORM'
    class_list = [d.name for d in os.scandir(sub_dir) if d.is_dir()]
    class_list = sorted(class_list)
    fold1_list ,fold2_list,fold3_list = [],[],[]
#    train_val_dict = {'ADI':235, 
#                      'BACK':235, 
#                      'DEB':208, 
#                      'LYM':200, 
#                      'MUC':200, 
#                      'MUS':209, 
#                      'NORM':205, 
#                      'STR':204, 
#                      'TUM':160}
    for class_name in class_list:
        
        fold1_list_tmp,fold2_list_tmp, fold3_list_tmp = get_list(os.path.join(sub_dir,class_name))
        print(class_name + " fold1 length is %d,fold2 length is %d,fold3 length is %d" \
              %(len(fold1_list_tmp),len(fold2_list_tmp),len(fold3_list_tmp)))
        fold1_list = fold1_list + fold1_list_tmp
        fold2_list = fold2_list + fold2_list_tmp
        fold3_list = fold3_list + fold3_list_tmp
        
    with open('fold1_train_list.db', 'wb') as fp:
        pickle.dump(fold1_list, fp)
    with open('fold2_train_list.db', 'wb') as fp:
        pickle.dump(fold2_list, fp)
    with open('fold3_train_list.db', 'wb') as fp:
        pickle.dump(fold3_list, fp)
        
