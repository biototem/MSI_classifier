#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 06:06:16 2021

@author: - Kwong
"""

import os
import glob
import random
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 13, 13

if __name__ == '__main__':
    sub_dir = '../DATA/NCT-CRC-HE-100K-NONORM'
    class_list = [d.name for d in os.scandir(sub_dir) if d.is_dir()]
    class_list = sorted(class_list)
    for i,class_name in enumerate(class_list):
        img_list = glob.glob(os.path.join(sub_dir,class_name,'*.tif'))
        sample_list = random.sample(img_list,9)
        for j,filepath  in enumerate(sample_list):
            img = io.imread(filepath)
            if j == 0:
                tmp_row_whole = img.copy()
            else:
                tmp_row_whole = np.hstack((tmp_row_whole,img))
        if i == 0:
            whole_img = tmp_row_whole.copy()
        else:
            whole_img = np.vstack((whole_img,tmp_row_whole))
            
    io.imsave('save.png',whole_img)