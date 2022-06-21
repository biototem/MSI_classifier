# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import albumentations
from albumentations.pytorch import ToTensorV2
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm as tqdm
import pickle
import glob
from model.densenet_ibn_b import densenet121_ibn_b
from utils.DataReader import MILdataset
from utils.train_utils import group_proba, calc_accuracy, group_argtopk, group_identify
from utils.nn_metrics import get_roc_best_threshold, get_mAP_best_threshold
from config import native_cfg as cfg

def main(parser):
    global args, best_acc
    args = parser.parse_args()

    # cnn
    model = densenet121_ibn_b(num_classes=2, pretrained = False)
    model = nn.DataParallel(model.cuda())
    # model.load_state_dict(torch.load("/media/totem_disk/totem/haosen/2020_07_18_5_NEW_TCGA_dense_ibn_b_512.pth")["state_dict"])
    
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1, args.weights/(1-args.weights)])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    cudnn.benchmark = True

    # normalization
    train_trans = albumentations.Compose([
        albumentations.VerticalFlip(),
        albumentations.HorizontalFlip(),
        albumentations.Rotate(360),
        albumentations.ShiftScaleRotate(),
        albumentations.HueSaturationValue(hue_shift_limit=3),
        albumentations.GaussianBlur(blur_limit=(1, 3)),
        albumentations.RandomScale(scale_limit=0.2),
        albumentations.Resize(cfg["img_hight"], cfg["img_width"]),
        ToTensorV2()
    ])
    val_trans = albumentations.Compose([albumentations.Resize(cfg["img_hight"], cfg["img_width"]),
                                        ToTensorV2()])
    
    with open(args.select_lib, 'rb') as fp:
        target_train_slide, target_val_slide = pickle.load(fp)
    # load data
    train_dset = MILdataset(args.train_lib, args.k, train_trans, args.train_dir,
                            data_expansion="average", select_condition=target_train_slide,
                            grid_mode="random_drop", grid_drop=0.8)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    val_dset = MILdataset(args.val_lib, args.k, val_trans,args.train_dir,
                          select_condition=target_val_slide)
    val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, 'checkpoint')):
        os.makedirs(os.path.join(args.output, 'checkpoint'))
    if not os.path.exists(os.path.join(args.output, 'train_infer_probs')):
        os.makedirs(os.path.join(args.output, 'train_infer_probs'))

    time_mark = time.strftime('%Y_%m_%d_',time.localtime(time.time()))
    # time_mark = '2020_03_06_'
    # 以当前时间作为保存的文件名标识

    # 用于存储每一轮算出来的top k index
    early_stop_count = 0
    # 标记是否early stop的变量，该变量>epochs*2/3时,就开始进行停止训练的判断
    list_save_dir = os.path.join(args.output, 'topk_list')
    if not os.path.isdir(list_save_dir): os.makedirs(list_save_dir)

    # 是否断点续训
    if args.resume:
        # 不指定从哪一轮开始，则默认为从最后一轮重新开始训练
        if args.resume_molde_path is None:
            path = glob.glob(os.path.join(args.output, 'checkpoint', "*.pth"))
            if len(path) == 0:
                raise ValueError("There is no weight file in checkpoint directory")
            path = sorted(path, key=lambda x: int(os.path.basename(x).split("_")[3]))
            checkpoint = torch.load(path[-1])
        else:
            checkpoint = torch.load(args.resume_molde_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_dset.load_state_list(checkpoint["dataset_state_list"])
        print("=> loaded checkpoint epoch: {}".format(start_epoch))
    else:
        # open output file
        fconv = open(os.path.join(args.output, time_mark + 'all_densenet121_ibn_CE.csv'), 'w',
                     encoding='utf-8-sig', newline='')
        writer = csv.writer(fconv)
        writer.writerow(["", "Train", "", "", "", "Train_whole", "", "",
                         "Validation", "", "", "", "Validation_whole"])
        writer.writerow(['epoch', 'train_precision', 'train_recall', 'train_f1', 'train_loss',
                         'true_precision', 'true_recall', 'true_f1',
                         'val_precision', 'val_recall', 'val_f1', 'val_loss', 'whole_AUC', "best_roc_auc_threshold",
                         'whole_mAP', "best_mAP_threshold",
                         'precision_0.3', 'recall_0.3', 'f1_0.3', 'precision_0.5', 'recall_0.5', 'f1_0.5',
                         'precision_0.7', 'recall_0.7', 'f1_0.7'])
        fconv.close()

        start_epoch = 1
        print("=> no checkpoint found")

    with open(os.path.join("/media/biototem/Elements/lisen/MSI_MSS_2020_8_12/repeat_TCGA-fold3_naive_512to224_average_fixed=0.2_lr=1e-5",
                           "dataset_status.pkl"), "rb") as fp:
        lib = pickle.load(fp)
        train_dset.load_state_list(lib)

    # loop throuh epochs
    for epoch in range(start_epoch, args.nepochs+1):
        if early_stop_count >= 10:
            print('Early stop at Epoch:' + str(epoch))
            break
        start_time = time.time()
        # Train
#         topk_exist_flag = False
#         if os.path.exists(os.path.join(list_save_dir, time_mark + '_224.pkl')) and epoch == 0:
#             with open(os.path.join(list_save_dir, time_mark + '_224.pkl'), 'rb') as fp:
#                 topk_list = pickle.load(fp)
#
# #            topk = topk_list[-1][0]
#             train_probs = topk_list[-1][1]
#             positive_topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[:, 1], args.k)
#             # negative_topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[:, 0], args.k)
#             topk = positive_topk
#             topk_exist_flag = True
#
#         else:
#             topk_list = []

        train_dset.setmode(1)
        train_probs = inference(epoch, train_loader, model, 'train')
        positive_topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[:, 1], args.k)
        # negative_topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[:, 0], args.k)
        topk = positive_topk
        # t_pred = group_max(train_probs,topk,args.k)

        # repeat = 0
        # if epoch >= 2/3*args.nepochs:
        #     # repeat = np.random.choice([3,5])
        #     # 前10轮设定在训练时复制采样,后10轮后随机决定是否复制采样
        #     if len(topk_list) > 0:
        #         topk_last = topk_list[-1][0]
        #         if sum(np.not_equal(topk_last, topk)) < 0.1 * len(topk):
        #             early_stop_count +=1
        # if not topk_exist_flag:
        #     topk_list.append((topk.copy(),train_probs.copy()))
        # with open(os.path.join(list_save_dir, time_mark + '_224.pkl'), 'wb') as fp:
        #     pickle.dump(topk_list, fp)

        train_dset.maketraindata(topk, repeat=0)
        train_dset.shuffletraindata()
        train_dset.setmode(2)
        train_whole_precision,train_whole_recall,train_whole_f1,train_whole_loss = train_predict(epoch, train_loader, model, criterion, optimizer,'train')
        print('\tTraining  Epoch: [{}/{}] Precision: {} Recall:{} F1score:{} Loss: {}'.format(epoch,
              args.nepochs, train_whole_precision,train_whole_recall,train_whole_f1,train_whole_loss))
        all_result = [epoch, train_whole_precision, train_whole_recall, train_whole_f1, train_whole_loss]

        # topk = group_argtopk(np.array(train_dset.slideIDX), train_probs[train_dset.label_mark], args.k)
        # t_pred = group_max(train_probs,topk,args.k)

        # train all
        train_pred = group_identify(train_dset.slideIDX, train_probs)
        metrics_meters = calc_accuracy(train_pred, train_dset.targets)
        all_result.extend([metrics_meters['precision'], metrics_meters['recall'], metrics_meters['f1score']])

        # val
        val_dset.setmode(1)
        val_whole_precision,val_whole_recall,val_whole_f1,val_whole_loss,val_probs = train_predict(epoch, val_loader, model, criterion, optimizer, 'val')
        all_result.extend([val_whole_precision, val_whole_recall, val_whole_f1, val_whole_loss])
        # v_topk = group_argtopk(np.array(val_dset.slideIDX), val_probs[val_dset.label_mark], args.k)
        # v_pred = group_max(val_probs,v_topk,args.k)
        # val_pred = group_identify(val_dset.slideIDX, val_probs)

        # val all
        msi_pro = group_proba(val_dset.slideIDX, val_probs, 0.5)
        roc_auc = roc_auc_score(val_dset.targets, msi_pro)
        best_roc_auc_threshold = get_roc_best_threshold(y_true=val_dset.targets,
                                                        y_score=msi_pro)
        mAP = average_precision_score(val_dset.targets, msi_pro)
        best_mAP_auc_threshold = get_mAP_best_threshold(y_true=val_dset.targets,
                                                        y_score=msi_pro)
        all_result.extend([round(roc_auc, 4), round(best_roc_auc_threshold, 5),
                           round(mAP, 4), round(best_mAP_auc_threshold, 5)])
        print('\tValidation  Epoch: [{}/{}] AUC: {} mAP:{} Loss: {}'.format(epoch,
              args.nepochs, roc_auc, mAP, val_whole_loss))

        for thres in [0.3, 0.5, 0.7]:
            v_pred = np.where(msi_pro > thres, 1, 0)
            metrics_meters = calc_accuracy(v_pred, val_dset.targets)
            all_result.extend([metrics_meters['precision'], metrics_meters['recall'], metrics_meters['f1score']])
            str_logs = ['{} - {:.4}'.format(k, v) for k, v in metrics_meters.items()]
            s = ', '.join(str_logs)
            print('\tAll Validation  Epoch: [{}/{}] threshold:{} '.format(epoch, args.nepochs, thres) + s)

        fconv = open(os.path.join(args.output, time_mark + 'all_densenet121_ibn_CE.csv'), 'a',
                     encoding='utf-8-sig', newline='')
        writer = csv.writer(fconv)
        writer.writerow(all_result)
        fconv.close()

        # Save best model
        # tmp_acc = val_whole_f1 #(metorics_meters['acc'] + metrics_meters['recall'])/2 #- metrics_meters['fnr']*args.weights
        # if tmp_acc >= best_acc:
        #     best_acc = tmp_acc.copy()
        #     best_metric_probs_inf_save['train_probs'] = train_probs.copy()
        #     best_metric_probs_inf_save['val_probs'] = val_probs.copy()
            
        if epoch > 0:
            eopch_save = {"state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch+1,
                          "dataset_state_list": [train_dset.slidenames, train_dset.targets,
                                                 train_dset.batch, train_dset.grid]}
            # result_excel_origin(train_dset,t_pred,time_mark + 'train_' + str(epoch+1),os.path.join(args.output, time_mark + 'result'))
            # result_excel_origin(val_dset,v_pred,time_mark + 'val_'+ str(epoch+1),os.path.join(args.output, time_mark + 'result'))

            np.save(os.path.join(args.output, 'train_infer_probs', time_mark + 'train_infer_probs_'
                                 + str(epoch+1) + '.npy'), train_probs)
            np.save(os.path.join(args.output, 'train_infer_probs', time_mark + 'val_infer_probs_' +
                                 str(epoch+1) + '.npy'), val_probs)
            torch.save(eopch_save,
                       os.path.join(args.output, 'checkpoint', time_mark + str(epoch + 1) + '_densenet_ibn_b_224.pth'))
                
        print('\tEpoch %d has been finished, needed %.2f sec.' % (epoch + 1,time.time() - start_time))                

        # torch.save(best_metric_probs_inf_save, os.path.join(args.output, time_mark +'best_metric_probs_inf_224.db'))
        # torch.save(eopch_save, os.path.join(args.output, time_mark +'densenet121_ibn_b_checkpoint_224.pth'))
    

def inference(epoch, loader, model, phase):
    model.eval()
    probs = np.zeros((1,2))
#    logs = {}
    whole_probably = 0.

    with torch.no_grad():
        with tqdm(loader, desc='Epoch:' + str(epoch) + ' ' + phase + '\'s inferencing',
                  file=sys.stdout, disable = False) as iterator:
            for i, (input, _) in enumerate(iterator):
                input = input.cuda()
                output = F.softmax(model(input), dim=1)
                prob = output.detach().clone()
                prob = prob.cpu().numpy()                
                batch_proba = np.mean(prob,axis=0)
                probs = np.row_stack((probs,prob))
                whole_probably = whole_probably + batch_proba

                iterator.set_postfix_str('batch proba :' + str(batch_proba))                                    
                
            whole_probably = whole_probably / (i+1)
            iterator.set_postfix_str('Whole average probably is ' + str(whole_probably))
            
    probs = np.delete(probs, 0, axis=0)
    return probs.reshape(-1,2)


def train_predict(epoch, loader, model, criterion, optimizer, phase='train',
                  grad_add=False, accumulation=2):
    if phase == 'val':
        model.eval()
        probs = np.zeros((1,2),np.float32)
    elif phase == 'train':
        model.train()
        logs = {}
    whole_loss = 0.
    whole_target = np.zeros((1,1),np.int8)
    whole_predict = np.zeros((1,1),np.int8)

    with tqdm(loader, desc='Epoch:' + str(epoch) + ' is ' + phase,
              file=sys.stdout, disable= False) as iterator:
        with torch.set_grad_enabled(phase == 'train'):
            for i, (input, target) in enumerate(iterator):
                input = input.cuda()                
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                _, pred = torch.max(output, 1)
                pred = np.int8(pred.data.cpu().numpy())
                target = target.cpu().numpy()
                whole_target = np.row_stack((whole_target,np.int8(target.reshape(-1,1))))
                whole_predict = np.row_stack((whole_predict,pred.reshape(-1,1)))
                # train的时候显示batch的metric,val的时候显示batch的平均probably
                if phase == 'train':
                    loss.backward()
                    # 增加梯度累加模式,默认为False;accumulation是梯度累积的batch倍数,默认为2倍. by Bohrium.Kwong 2020.05.18
                    if grad_add:
                        if (i+1)%accumulation == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    metrics_meters = calc_accuracy(pred, target)
                    logs.update(metrics_meters)
                    logs.update({'loss':loss.item()})
                    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
                    s = ', '.join(str_logs)
                    
                else:
                    output = F.softmax(output, dim=1)
                    prob = output.detach().clone()
                    prob = prob.cpu().numpy()
                    probs = np.row_stack((probs,prob))
                    batch_proba = np.mean(prob,axis=0)
                    s = 'batch proba :' + str(batch_proba)
                iterator.set_postfix_str(s)
                whole_loss += loss.item()
            
    whole_target = np.delete(whole_target, 0, axis=0)
    whole_predict = np.delete(whole_predict, 0, axis=0)
    whole_metrics_meters = calc_accuracy(whole_predict, whole_target)
    
    if phase == 'train':
        return whole_metrics_meters['precision'],whole_metrics_meters['recall'],whole_metrics_meters['f1score'],round(whole_loss/(i+1),3)
    else:
        probs = np.delete(probs, 0, axis=0)
        return whole_metrics_meters['precision'],whole_metrics_meters['recall'],whole_metrics_meters['f1score'],round(whole_loss/(i+1), 3),probs.reshape(-1,2)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
    parser.add_argument('--train_lib', type=str,
                        default='/Your_dir/new_TCGA_data_lib.db',
                        help='path to train MIL library binary')
    parser.add_argument('--select_lib', type=str,
                        default="/Your_dir/spilt_new_TCGA_fold3_3.db",
                        help='path to validation MIL library binary. If present.')
    parser.add_argument('--train_dir', type=str,
                        default='/Your_orgin_dataset_dir')
    parser.add_argument('--val_lib', type=str,
                        default='/media/totem_disk/totem/haosen/MSI_MSS_LiSen/data/new_TCGA_data_lib.db',
                        help='path to train MIL library binary')
    parser.add_argument('--output', type=str,
                        default='/Your_output_dir',
                        help='name of output file')
    parser.add_argument('--resume', type=bool, default=False, help='is the model retrained from the breakpoint')
    parser.add_argument('--resume_molde_path', type=str,
                        default=None,
                        help='The default is none, which means retraining from the last round. You can also specify the power down file path for the model, indicating which round to start from')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
    parser.add_argument('--nepochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
    # 如果是在docker中运行时需注意,因为容器设定的shm内存不够会出现相关报错,此时将num_workers设为0则可
    #parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
    parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
    parser.add_argument('--k', default=20, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')
    #parser.add_argument('--tqdm_visible',default = True, type=bool,help='keep the processing of tqdm visible or not, default: True')

    main(parser)

