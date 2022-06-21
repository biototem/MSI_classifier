# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import pickle
import cv2
import torch
import shutil
from tqdm import tqdm
import torch.utils.data as data


def load_data(libraryfile, select_condition=None):
    if not os.path.exists(libraryfile):
        raise ValueError("The file address does not exist:{}".format(libraryfile))
    with open(libraryfile, 'rb') as fp:
        lib = pickle.load(fp)

    if select_condition is None:
        slides_list = lib['slides']
        grid_list = lib['grid']
        targets_list = lib['targets']
        batch_list = lib['batch']
        name_list = lib["name"] if lib.get("name", None) is not None else None
        mask = []
        label_mask = []
        for i, g in enumerate(grid_list):
            mask.extend([True] * len(g))
            label_mask.append([lib['targets'][i]] * len(g))
    else:
        slides_list = []
        grid_list = []
        targets_list = []
        batch_list = []
        mask = []
        label_mask = []
        name_list = []
        if select_condition[0] == "slides" and isinstance(select_condition[1], list):
            for i, slide in enumerate(lib['slides']):
                # if slide in select_condition[1]:
                if slide in select_condition[1] or slide.split(" ")[0] in select_condition[1]\
                        or slide.split("-")[0] in select_condition[1]:
                    slides_list.append(slide)
                    grid_list.append(lib['grid'][i])
                    targets_list.append(lib['targets'][i])
                    batch_list.append(lib['batch'][i])
                    name_list.append(lib["name"][i] if lib.get("name", None) is not None else None)
                    mask.extend([True] * len(grid_list[-1]))
                    label_mask.append([lib['targets'][i]] * len(grid_list[-1]))
                else:
                    mask.extend([False] * len(lib['grid'][i]))

        elif select_condition[0] == "batch" and isinstance(select_condition[1], list):
            for i, batch in enumerate(lib['batch']):
                if batch in select_condition[1]:
                    slides_list.append(lib['slides'][i])
                    grid_list.append(lib['grid'][i])
                    targets_list.append(lib['targets'][i])
                    batch_list.append(lib['batch'][i])
                    name_list.append(lib["name"][i] if lib.get("name", None) is not None else None)
                    mask.extend([True] * len(grid_list[-1]))
                    label_mask.append([lib['targets'][i]] * len(grid_list[-1]))
                else:
                    mask.extend([False] * len(lib['grid'][i]))

        if not any(name_list):
            name_list = None

    return slides_list, grid_list, targets_list, batch_list, name_list, mask, label_mask


class MILdataset(data.Dataset):
    """
    MIL和Naive的数据读取器,兼容了两者
    """
    def __init__(self, libraryfile='', min_data_len=0, transform=None, image_save_dir='', select_condition=None,
                 data_balance=False, grid_mode="no_drop", grid_drop=0, drop_limit=0):
        # 加载数据
        self.slidenames, self.grid, self.targets, \
        self.batch, self.name_list, _, self.label_mask = load_data(libraryfile=libraryfile,
                                                                   select_condition=select_condition)
        print('Number of tiles: {}'.format(len(self.grid)))
        self.min_data_len = min_data_len
        self.transform = transform
        self.mode = "train"
        self.image_save_dir = image_save_dir
        self._grid_drop = grid_drop
        self._data_balance = data_balance
        self._grid_mode = grid_mode
        self._drop_limit = drop_limit

        self.patch_static = {}
        self.classes = {}
        self.patch_static["origin"] = self._get_data_max_average_len()
        self.classes["origin"] = self._static_class_num()
        self.DataExpansion()

    def _get_data_max_average_len(self):
        slide_len = [len(i) for i in self.grid]
        patch = {"max": np.max(slide_len), "min": np.min(slide_len),
                 "mean": np.round(np.mean(slide_len), 1), "median": np.median(slide_len),
                 "patch_len": slide_len}
        return patch

    def _static_class_num(self):
        # # 统计每个slide的patch数量
        grid_len = [len(i) for i in self.grid]

        # 统计每个类别slide数量和patch数量
        classes = set(self.targets)
        classes_num = {}
        for key in classes:
            classes_num[key] = {
                "slide": self.targets.count(key),
                "patch": sum([i for index, i in enumerate(grid_len) if self.targets[index] == key])}
        classes_num["grid_len"] = grid_len
        return classes_num

    def load_state_list(self, state_list):
        # 保证模型断点续训
        self.slidenames = state_list[0]
        self.targets = state_list[1]
        self.batch = state_list[2]
        self.grid = state_list[3]
        mode = self._grid_mode
        self._grid_mode = "no_drop"
        self.getSlidexIDX()
        self._grid_mode = mode

    def getSlidexIDX(self):
        self.itera_grid = []
        self.slideIDX = []
        self.label_mark = []
        # slideIDX列表存放每个WSI以及其坐标列表的标记,假设有0,1,2号三个WSI图像,分别于grid中记录4,7,3组提取的坐标,\
        # 返回为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        for i, g in enumerate(self.grid):
            # fixed_drop是固定采样, 直接对对每个slide采len(grid)*(1-grid_drop)的样本数
            if self._grid_mode == "fixed_drop" and self._grid_drop > 0:
                # 当大于最小丢弃数量才最进行丢弃
                if len(g) > self._drop_limit:
                    g = random.sample(g,
                                      int(len(g) * (1.0 - self._grid_drop)) if int(
                                          len(g) * (1.0 - self._grid_drop)) > 0 else 1)
                    # 当前slide已有的grid数量在k之下时,就进行随机重复采样
                    if len(g) < self.min_data_len:
                        g = g + [(g[x]) for x in np.random.choice(range(len(g)), self.min_data_len - len(g))]
                self.grid[i] = g

            self.slideIDX.extend([i] * len(g))
            self.itera_grid.extend(g)
            if isinstance(self.targets[i], int) and int(self.targets[i]) == 0:
                self.label_mark.extend([(True, False)]*len(g))
            elif isinstance(self.targets[i], int) and int(self.targets[i]) == 1:
                self.label_mark.extend([(False, True)]*len(g))

        self.patch_static["data_aug"] = self._get_data_max_average_len()
        self.classes["data_aug"] = self._static_class_num()

    def DataExpansion(self):
        if self._data_balance:
            classes = list(set(self.targets))
            class_gradnum_dict = dict(zip(classes, [sum([len(grid) for i, grid in enumerate(self.grid) if
                                                    self.targets[i] == label]) for label in classes]))
            num_to_class_dict = dict(zip(class_gradnum_dict.values(), class_gradnum_dict.keys()))
            minority_class_gridnum = min(list(class_gradnum_dict.values()))
            majority_class_gridnum = max(list(class_gradnum_dict.values()))
            minority_class = num_to_class_dict[minority_class_gridnum]
            new_slide = []
            new_grid = []
            while True:
                index = random.randint(0, len(self.slidenames) - 1)
                if self.targets[index] == minority_class:
                    new_slide.append(self.slidenames[index])
                    grid_index = list(range(len(self.grid[index])))
                    random.shuffle(grid_index)
                    grid_index = grid_index[:len(grid_index)//2]
                    grid = [self.grid[index][i] for i in grid_index]
                    new_grid.append(grid)
                    minority_class_gridnum += len(new_grid[-1])
                    if minority_class_gridnum * 1.0 / majority_class_gridnum >= 0.95:
                        break
            self.targets.extend([minority_class] * int(len(new_slide)))
            self.batch.extend([self.batch[self.slidenames.index(name)] for name in new_slide])
            self.grid.extend(new_grid)
            self.slidenames.extend(new_slide)

        self.getSlidexIDX()

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs=None):
        if idxs is None:
            lst = range(len(self.itera_grid))
        else:
            lst = idxs

        self.t_data = [(self.batch[self.slideIDX[x]], self.slideIDX[x], self.itera_grid[x],
                        self.targets[self.slideIDX[x]], 0, 0) for x in lst]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == "val":
            # 为预测时使用，会从所有WSI文件中返回全部的region的图像
            slideIDX = self.slideIDX[index]
            (k, j) = self.itera_grid[index]
            target = self.targets[slideIDX]

            img = cv2.imread(os.path.join(self.image_save_dir,  self.batch[slideIDX], self.slidenames[slideIDX],
                             self.slidenames[slideIDX] + '_' + str(k) + '_' + str(j) + '.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                img = self.transform(image=img)["image"]
            return img.float().div(255), target

        elif self.mode == "train":
            # 为训练时使用，
            # 只会根据指定的index(经过上一轮MIL过程得出)
            # 从全部WSI文件中筛选对应的坐标列表,返回相应的训练图像和label
            batch, slideIDX, (k, j), target, h_value, radius = self.t_data[index]
            img = cv2.imread(os.path.join(self.image_save_dir, batch, self.slidenames[slideIDX],
                                          self.slidenames[slideIDX] + '_' + str(k) + '_' + str(j) + '.jpg'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            return img.float().div(255), target

        elif self.mode == "path":
            # 返回数据集路径
            slideIDX = self.slideIDX[index]
            (k, j) = self.itera_grid[index]
            return os.path.join(self.image_save_dir,  self.batch[slideIDX], self.slidenames[slideIDX],
                                self.slidenames[slideIDX] + '_' + str(k) + '_' + str(j) + '.jpg')

        elif self.mode == "feature":
            name = self.slidenames[index]
            img_path = [os.path.join(self.image_save_dir, self.batch[index], self.slidenames[index],
                                     self.slidenames[index] + '_' + str(k) + '_' + str(j) + '.jpg')
                        for (k, j) in self.grid[index]]

            return (name, img_path)

    def __len__(self):
        if self.mode == "val" or self.mode == "path":
            return len(self.itera_grid)
        elif self.mode == "train":
            return len(self.t_data)
        elif self.mode == "feature":
            return len(self.slidenames)


class FeatureDataset(data.Dataset):
    """
    用于特征训练的数据读取器
    """

    def __init__(self, feature_dir, input_depth, libraryfile, select_condition=None,
                 data_balance=False, min_data_length=None, connect_feature=False, sort=False, save_dir=None):
        # 加载数据
        self.slidenames, self.grid, self.targets, \
        self.batch, self.name_list, _, self.label_mask = load_data(libraryfile=libraryfile,
                                                                   select_condition=select_condition)
        if self.name_list is None:
            self.name_list = self.slidenames
        print('Number of tiles: {}'.format(len(self.grid)))
        self.feature_dir = feature_dir
        self.input_depth = input_depth
        self._data_balance = data_balance
        # self.padding_method = padding_method
        # self.max_padding_length = max_padding_length
        # self.padding_data = padding_data
        self.min_data_length = min_data_length
        self.connect_feature = connect_feature
        self.sort = sort
        self.save_dir = save_dir
        self.patch_static = {}
        self.classes = {}
        self.patch_static["origin"] = self._get_data_max_average_len()
        self.classes["origin"] = self._static_class_num()
        self.DataExpansion()
        self.sortList()

    def _get_data_max_average_len(self):
        slide_len = [len(i) for i in self.grid]
        patch = {"max": np.max(slide_len), "min": np.min(slide_len),
                 "mean": np.round(np.mean(slide_len), 1), "median": np.median(slide_len),
                 "patch_len": slide_len}
        return patch

    def _static_class_num(self):
        # # 统计每个slide的patch数量
        grid_len = [len(i) for i in self.grid]

        # 统计每个类别slide数量和patch数量
        classes = set(self.targets)
        classes_num = {}
        for key in classes:
            classes_num[key] = {
                "slide": self.targets.count(key),
                "patch": sum([i for index, i in enumerate(grid_len) if self.targets[index] == key])}
        classes_num["grid_len"] = grid_len
        return classes_num

    def load_state_list(self, state_list):
        # 保证模型断点续训
        self.slidenames = state_list[0]
        self.targets = state_list[1]
        self.batch = state_list[2]
        self.grid = state_list[3]

    def DataExpansion(self):
        # 是否进行数据平衡
        if self._data_balance:
            classes = list(set(self.targets))
            class_gradnum_dict = dict(zip(classes, [sum([len(grid) for i, grid in enumerate(self.grid) if
                                                    self.targets[i] == label]) for label in classes]))
            num_to_class_dict = dict(zip(class_gradnum_dict.values(), class_gradnum_dict.keys()))
            minority_class_gridnum = min(list(class_gradnum_dict.values()))
            majority_class_gridnum = max(list(class_gradnum_dict.values()))
            minority_class = num_to_class_dict[minority_class_gridnum]
            new_slide = []
            new_grid = []
            while True:
                index = random.randint(0, len(self.slidenames) - 1)
                if self.targets[index] == minority_class:
                    new_slide.append(self.slidenames[index])
                    grid_index = list(range(len(self.grid[index])))
                    random.shuffle(grid_index)
                    grid_index = grid_index[:len(grid_index)//2]
                    grid = [self.grid[index][i] for i in grid_index]
                    new_grid.append(grid)
                    minority_class_gridnum += len(new_grid[-1])
                    if minority_class_gridnum * 1.0 / majority_class_gridnum >= 0.95:
                        break
            self.targets.extend([minority_class] * int(len(new_slide)))
            self.batch.extend([self.batch[self.slidenames.index(name)] for name in new_slide])
            self.grid.extend(new_grid)
            self.slidenames.extend(new_slide)

        # # 是否对特征数据进行填充
        # if self.padding_method in ["max", "mean", "set"]:
        #     if self.padding_method == "max":
        #         max_padding_len = int(self.patch_static["origin"]["max"])
        #     elif self.padding_method == "mean":
        #         max_padding_len = int(self.patch_static["origin"]["mean"])
        #     elif self.padding_method == "set":
        #         max_padding_len = self.max_padding_length
        #
        #     if self.padding_data == "origin":
        #         for i, g in enumerate(self.grid):
        #             g = random.sample(g, max_padding_len) if max_padding_len <= len(g) else \
        #                 random.sample(g, max_padding_len%len(g)) + g * int(max_padding_len//len(g))
        #             self.grid[i] = g
        #     elif self.padding_method == "zero":
        #         for i, g in enumerate(self.grid):
        #             g = random.sample(g, max_padding_len) if max_padding_len <= len(g) else \
        #                 [(-1, -1)] * int(max_padding_len - len(g)) + g
        #             self.grid[i] = g

        # 保证bag内instance数量一定大于等于最小阈值
        for i, g in enumerate(self.grid):
            if len(g) < self.min_data_length:
                g = random.sample(g, self.min_data_length%len(g)) + g * int(self.min_data_length//len(g))
                self.grid[i] = g

        self.patch_static["data_aug"] = self._get_data_max_average_len()
        self.classes["data_aug"] = self._static_class_num()

        # 将instance聚合保存成bag，加速读取速度
        if self.connect_feature:
            # 创建特征保存目录
            if self.save_dir is None:
                self.save_dir = os.path.join(self.feature_dir, "tmp")

            # 清空目录
            if not os.path.exists(self.save_dir):
                # shutil.rmtree(save_path)
                os.mkdir(self.save_dir)

            # # 将instance聚合保存成bag保存起来. 加快读取速度
            # with tqdm(self.grid, desc="Connect Feature") as iterator:
            #     for index, grid in enumerate(iterator):
            #         slidename = self.slidenames[index]
            #         name = self.name_list[index]
            #         feature = [np.load(os.path.join(self.feature_dir, self.batch[index], slidename,
            #                                         slidename + '_' + str(k) + '_' + str(
            #                                             j) + '.npy')) if k != -1 and j != -1
            #                    else np.zeros((1, self.input_depth))
            #                    for (k, j) in grid]
            #         feature = np.ascontiguousarray(np.concatenate(feature, axis=0))
            #         np.save(os.path.join(self.save_dir, "{}.npy").format(name), feature)

    def sortList(self):
        if self.sort:
            length = self.patch_static["data_aug"]["patch_len"]
            tmp = [(length[i], self.slidenames[i], self.grid[i], self.targets[i],
                    self.batch[i], self.name_list[i]) for i in range(len(self.slidenames))]
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)
            self.slidenames = [i[1] for i in tmp]
            self.grid = [i[2] for i in tmp]
            self.targets = [i[3] for i in tmp]
            self.batch = [i[4] for i in tmp]
            self.name_list = [i[5] for i in tmp]

    def __getitem__(self, index):
        target = self.targets[index]

        if not self.connect_feature:
            grid = self.grid[index]
            feature = [np.load(os.path.join(self.feature_dir,  self.batch[index], self.slidenames[index],
                               self.slidenames[index] + '_' + str(k) + '_' + str(j) + '.npy')) if k!=-1 and j!=-1
                       else np.zeros((1, self.input_depth))
                       for (k, j) in grid]
            feature = np.ascontiguousarray(np.concatenate(feature, axis=0))
        else:

            feature = np.load(os.path.join(self.save_dir, "{}.npy").format(self.name_list[index]))

        # numpy to tensor
        feature = torch.from_numpy(feature)
        return feature, target

    def __len__(self):
        return len(self.grid)
