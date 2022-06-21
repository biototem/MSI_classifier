import os
import sys
sys.path.append("/")
import numpy as np
import csv
import argparse
import glob
import pickle

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model.densenet_ibn_b import densenet121_ibn_b
from utils.DataReader import MILdataset
from utils.train_utils import group_proba, calc_accuracy
from naive_train import inference
from utils.nn_metrics import get_all_metrics
from utils.plot_utils import plot_confusion_matrix, plot_roc_auc, plot_precision_recall
import pandas as pd
import albumentations
from albumentations.pytorch import ToTensorV2
from model.model_utils import get_pretrained_model


class Prediction(object):
    def __init__(self, args,  select_condition=None, labels=[0, 1]):
        self.libraryfile = args.data_lib
        self.data_dir = args.data_dir
        self.model_weights = args.model_weights
        self.threshold = args.threshold
        self.labels = labels
        self.img_height = args.img_height
        self.img_width = args.img_width
        self.batch_size = args.batch_size
        self.workers = args.workers

        if args.output_dir is None or len(args.output_dir) == 0:
            self.output_dir = os.path.join(os.path.dirname(__file__), "", "tmp")
        else:
            self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.visual_save_dir = os.path.join(self.output_dir, "output")
        if not os.path.exists(self.visual_save_dir):
            os.makedirs(self.visual_save_dir)
        self.select_condition = select_condition
        self.judge_model_weight_path()

        # 定义模型
        # self.model = densenet121_ibn_b(num_classes=2, pretrained=False)
        # self.model = nn.DataParallel(self.model)

        self.model = get_pretrained_model(model_name="tf_efficientnet_b7_ns",
                                          model_weight_path=None,
                                          num_classes=2,
                                          img_height=self.img_height,
                                          img_width=self.img_hight,
                                          pretrained=False,
                                          verbose=True)

    def judge_model_weight_path(self):
        if isinstance(self.model_weights, list) and len(self.model_weights) > 0:
            self.model_weights_path = self.model_weights
        elif isinstance(self.model_weights, str) and os.path.isfile(self.model_weights):
            self.model_weights_path = [self.model_weights]
        elif isinstance(self.model_weights, str) and os.path.isdir(self.model_weights):
            self.model_weights_path = glob.glob(os.path.join(self.model_weights, "*.pth"))
            # self.model_weights_path = sorted(self.model_weights_path, key=lambda x: int(os.path.basename(x).split("_")[3]))
        else:
            raise ValueError("model_weights set error!")

    def get_statistics_title(self, k_fold=False):
        statistics_title = []
        statistics_title.append("slide id")
        statistics_title.append("slide patch size")
        statistics_title.append("negative probability")
        statistics_title.append("positive probability")
        statistics_title.append("predicted value")
        statistics_title.append("true value")

        if k_fold:
            statistics_title.append("")
            statistics_title.append("frist positive probability")
            statistics_title.append("second positive probability")
            statistics_title.append("third positive probability")
        return statistics_title

    def load_model(self, model_weight):
        # self.model = densenet121_ibn_b(num_classes=2, pretrained=False)
        # self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(model_weight)["state_dict"])
        self.model.cuda().eval()

    def predict(self):
        trans = albumentations.Compose([albumentations.Resize(self.img_hight, self.img_width),
                                        ToTensorV2()])
        dset = MILdataset(self.libraryfile, 0, trans, self.data_dir,
                          select_condition=self.select_condition)
        loader = DataLoader(
            dset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=False)
        dset.setmode("val")

        k_positive_prob_list = []
        k_outcome_list = []
        for i, model_weights in enumerate(self.model_weights_path):
            self.load_model(model_weights)
            # 预测
            pred_prob = inference(i, loader, self.model, "inferince")
            positive_prob = group_proba(dset.slideIDX, pred_prob, 0.5)
            k_outcome_list.append(np.where(positive_prob > 0.5, 1, 0))
            k_positive_prob_list.append(np.expand_dims(positive_prob, axis=1))
            with open(os.path.join(self.visual_save_dir, "val_patch_probs.pkl"), "wb") as fp:
                pickle.dump([dset.slideIDX, pred_prob], fp)

        if len(k_positive_prob_list) == 1:
            finally_positive_prob = np.squeeze(k_positive_prob_list[0])
            finally_outcome = np.where(finally_positive_prob >= self.threshold, 1, 0)
        else:
            finally_positive_prob = np.mean(np.concatenate(k_positive_prob_list, axis=-1), axis=-1, keepdims=False)
            finally_outcome = np.where(finally_positive_prob >= self.threshold, 1, 0)

            for i in range(len(k_positive_prob_list)):
                plot_roc_auc(y_true=dset.targets,
                             y_score=k_positive_prob_list[i],
                             tiltle_suffix="-fold_{}".format(i+1),
                             save_dir=self.visual_save_dir,
                             is_show=False)
                plot_confusion_matrix(y_true=dset.targets,
                                      y_pred=k_outcome_list[i],
                                      labels=self.labels,
                                      title="confusion matrix-fold_{}".format(i+1),
                                      save_dir=self.visual_save_dir,
                                      is_show=False)
        np.save(os.path.join(self.visual_save_dir, "val_probs"), finally_positive_prob)

        acc, precision, recall, f1_score, auc,\
        average_precision, best_roc_auc_threshold, best_mAP_auc_threshold = get_all_metrics(
            positive_prob=finally_positive_prob,
            outcome=finally_outcome,
            label=dset.targets)

        plot_roc_auc(y_true=dset.targets,
                     y_score=finally_positive_prob,
                     save_dir=self.visual_save_dir,
                     is_show=False)
        plot_precision_recall(y_true=dset.targets,
                              y_score=finally_positive_prob,
                              save_dir=self.visual_save_dir,
                              is_show=False)
        plot_confusion_matrix(y_true=dset.targets,
                              y_pred=finally_outcome,
                              labels=self.labels,
                              title="confusion matrix",
                              save_dir=self.visual_save_dir,
                              is_show=False)
        plot_confusion_matrix(y_true=dset.targets,
                              y_pred=finally_outcome,
                              labels=self.labels,
                              title="confusion matrix_percentage",
                              save_dir=self.visual_save_dir,
                              is_show=False,
                              is_percentage=True)

        fp_0 = open(os.path.join(self.visual_save_dir, 'statistics.csv'), mode="w",
                    encoding='utf-8-sig', newline='')
        writer_0 = csv.writer(fp_0)
        writer_0.writerow(["acc", "precision", "recall", "f1_score", "roc_auc",
                           "best_roc_auc_threshold", "average_precision", "best_mAP_threshold"])
        writer_0.writerow([round(acc, 4), round(precision, 4), round(recall, 4),
                           round(f1_score, 4), round(auc, 4), round(best_roc_auc_threshold, 4),
                           round(average_precision, 4), round(best_mAP_auc_threshold, 4)])
        writer_0.writerow("\n")
        if len(self.model_weights_path)>1:
            writer_0.writerow(["acc_flod", "precision_flod", "recall_flod", "f1_score_flod",
                               "roc_auc_flod", "average_precision_flod"])
            for i in range(len(k_outcome_list)):
                acc_flod , precision_flod , recall_flod , f1_score_flod , auc_flod , \
                average_precision_flod, best_roc_auc_threshold_flod = get_all_metrics(
                    positive_prob=k_positive_prob_list[i],
                    outcome=k_outcome_list[i],
                    label=dset.targets)

                writer_0.writerow([round(acc_flod, 4), round(precision_flod, 4), round(recall_flod, 4),
                                   round(f1_score_flod, 4), round(auc_flod, 4), round(average_precision_flod, 4),
                                   best_roc_auc_threshold_flod])
            writer_0.writerow("\n")

            csv_title = self.get_statistics_title(True)
            writer_0.writerow(csv_title)
            for i in range(len(dset.slidenames)):
                writer_0.writerow([dset.slidenames[i], dset.classes["origin"]["grid_len"][i], 1-np.round(finally_positive_prob[i], 4),
                                   np.round(finally_positive_prob[i], 4), finally_outcome[i], dset.targets[i],
                                  ""]+[np.round(k_positive_prob_list[j][i], 4) for j in range(len(k_positive_prob_list))])
        else:
            csv_title = self.get_statistics_title(False)
            writer_0.writerow(csv_title)
            for i in range(len(dset.slidenames)):
                # if finally_outcome[i] != dset.targets[i]:
                    writer_0.writerow([dset.slidenames[i], dset.classes["origin"]["grid_len"][i], 1-np.round(finally_positive_prob[i], 4),
                                       np.round(finally_positive_prob[i], 4), finally_outcome[i], dset.targets[i]])
        fp_0.close()

    def inference(self):
        trans = albumentations.Compose([albumentations.Resize(self.img_hight, self.img_width),
                                        ToTensorV2()])
        dset = MILdataset(self.libraryfile, 0, trans, self.data_dir,
                          select_condition=self.select_condition)
        loader = DataLoader(
            dset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=False)
        dset.setmode("test")

        k_positive_prob_list = []
        for model_weights in self.model_weights_path:
            self.load_model(model_weights)
            # 预测
            pred_prob = inference(0, loader, self.model, "inferince")
            positive_prob = group_proba(dset.slideIDX, pred_prob, 0.5)
            k_positive_prob_list.append(np.expand_dims(positive_prob, axis=1))

        if len(k_positive_prob_list) == 1:
            finally_positive_prob = np.squeeze(k_positive_prob_list[0])
        else:
            finally_positive_prob = np.mean(np.concatenate(k_positive_prob_list, axis=-1), axis=-1, keepdims=False)
        finally_outcome = (np.where(finally_positive_prob >= self.threshold, 1, 0)).tolist()
        np.save(os.path.join(self.visual_save_dir, "test_probs"), finally_positive_prob)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='OkWin Model Prediction')
    parser.add_argument('--data_lib', type=str,
                        default="Your_dir/batch_data.db",
                        help='path to train MIL library binary')
    parser.add_argument('--select_lib', type=str,
                        default="Your_dir/select_batch.db",
                        help='path to validation MIL library binary. If present.')
    parser.add_argument('--data_dir', type=str,
                        default='Your_data_dir/')
    parser.add_argument('--output_dir', type=str,
                        default='/Your_output_dir',
                        help='name of output dir')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 512)')
    parser.add_argument('--img_width', type=int, default=224, help='the width of image (default: 224)')
    parser.add_argument('--img_height', type=int, default=224, help='the hight of image (default: 224)')
    parser.add_argument('--model_weights',
                        default="/Your_checkpoint_dir/densenet_ibn_b_224.pth",
                        type=str,
                        help='model weights file')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='threshold of classification judgment')
    parser.add_argument('--k', default=0, type=int,
                        help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    # 是否对数据集进行筛选
    if args.select_lib is not None:
        if os.path.isfile(args.select_lib):
            with open(args.select_lib, 'rb') as fp:
                select_condition = pickle.load(fp)
                select_condition = ['slides', select_condition[-1]]
        else:
            select_condition = args.select_lib
    else:
        select_condition = None
    predict = Prediction(args,
                         select_condition=select_condition,
                         labels=[0, 1])
    predict.predict()
