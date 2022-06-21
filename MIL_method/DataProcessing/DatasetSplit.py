import os
import pickle
import random
from utils.DataReader import load_data


def split_dataset(libraryfile, train_percent=0.9, shuffle=True, select_condition=None):
    """
    根据比例划分划分训练集和验证集
    """
    slides_list, grid_list, targets_list, batch_list, mask, label_mask = load_data(libraryfile, select_condition)
    samples = list(zip(slides_list, targets_list))
    classes = set(targets_list)

    if shuffle:
        random.shuffle(samples)

    # 获取每个类别的img索引
    class_index = dict([(k, []) for k in classes])
    for index, (slide, idx) in enumerate(samples):
        class_index[idx].append(index)

    # 打乱每个类别的img索引
    for k, v in class_index.items():
        random.shuffle(v)

    # 分割数据集
    dataset = {"train": [], "val": []}
    for k, v in class_index.items():
        dataset["train"].extend([os.path.basename(samples[i][0])
                                 for i in class_index[k][:int(len(class_index[k])*train_percent)]])
        dataset["val"].extend([os.path.basename(samples[i][0])
                               for i in class_index[k][int(len(class_index[k])*train_percent):]])
    return dataset

