# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda import amp
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import csv
import glob
import pickle
from torch.utils.data import DataLoader
import albumentations
from albumentations.pytorch import ToTensorV2
from model.lossfunction import CEAddL1Loss
from utils.DataReader import MILdataset
from model.densenet_ibn_b import densenet121_ibn_b
from utils.train_utils import group_proba, calc_accuracy
from utils.nn_metrics import get_roc_best_threshold, get_mAP_best_threshold
from torchsummary import summary
from model.model_utils import get_pretrained_model
import warnings
warnings.filterwarnings("ignore")


def main(args, cfg):
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

    # 是否对数据集进行筛选
    if isinstance(args.select_lib, str) and os.path.exists(args.select_lib):
        with open(args.select_lib, 'rb') as fp:
            target_train_slide, target_val_slide = pickle.load(fp)
            target_train_slide = ['slides', target_train_slide]
            target_val_slide = ['slides', target_val_slide]
    elif isinstance(args.select_lib, list) and len(args.select_lib) == 2:
        target_train_slide, target_val_slide = args.select_lib[0], args.select_lib[1]
    else:
        target_train_slide, target_val_slide = None, None

    # load data
    train_dset = MILdataset(args.train_lib, 0, train_trans, args.train_dir,
                            data_balance=cfg["data_balance"], select_condition=target_train_slide,
                            grid_mode=cfg["grid_mode"], grid_drop=cfg["grid_drop"],
                            drop_limit=cfg["drop_limit"])
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    val_dset = MILdataset(args.val_lib, 0, val_trans, args.train_dir, select_condition=target_val_slide)
    val_loader = DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    # 设置数据集状态
    train_dset.setmode("train")
    train_dset.maketraindata()
    val_dset.setmode("val")
    val_dset.maketraindata()

    # 加载模型
    # model = densenet121_ibn_b(num_classes=cfg["num_classes"], pretrained=False)
    # with torch.no_grad():
    #     summary(model.eval().cuda(), input_size=(3, 224, 224), device="cuda")
    # model = nn.DataParallel(model.cuda())
    
    model = get_pretrained_model(model_name="tf_efficientnet_b7_ns",
                                 model_weight_path=None,
                                 num_classes=2,
                                 img_height=cfg["img_height"],
                                 img_width=cfg["img_width"],
                                 pretrained=False,
                                 verbose=True)
    model.cuda()

    # 损失函数
    # 获取加权系数
    class_allnum = sum([i["patch"] for i in train_dset.classes["data_aug"].values() if isinstance(i, dict)])
    w = [round(train_dset.classes["data_aug"][1]["patch"]*1.0 / class_allnum, 2),
         round(train_dset.classes["data_aug"][0]["patch"]*1.0 / class_allnum, 2)]
    w = torch.Tensor(w)
    # criterion = nn.CrossEntropyLoss(w).cuda()     # BCE
    criterion = CEAddL1Loss(weight=w, alpha=1.0, beta=1.0).cuda()     # BCE+L1Loss
    optimizer = optim.Adam([{"params": model.parameters(), "initial_lr": cfg["lr"]}],
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2,
                                                               eta_min=cfg["min_lr"], last_epoch=-1)
    cudnn.benchmark = True

    # open output file
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output, 'checkpoint')):
        os.makedirs(os.path.join(args.output, 'checkpoint'))

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
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        train_dset.load_state_list(checkpoint["dataset_state_list"])
        train_dset.maketraindata()
        time_mark = os.path.basename(glob.glob(os.path.join(args.output, '*metrics.csv'))[0])
        time_mark = "_".join(time_mark.split("_")[0:3])
        print("=> loaded checkpoint epoch: {}".format(start_epoch))
    else:
        # time_mark = '2020_07_03_' 以当前时间作为保存的文件名标识
        time_mark = time.strftime('%Y_%m_%d', time.localtime(time.time()))
        fconv = open(os.path.join(args.output, time_mark + '_metrics.csv'), 'w',
                     encoding='utf-8-sig', newline='')
        writer = csv.writer(fconv)
        writer.writerow(["", "Train", "", "", "", "Validation", "", "", "", "Validation_whole"])
        writer.writerow(['epoch', 'train_precision', 'train_recall', 'train_f1', 'train_loss',
                         'val_precision', 'val_recall', 'val_f1', 'val_loss', 'whole_AUC', "best_roc_auc_threshold",
                         'whole_mAP', "best_mAP_threshold",
                         'precision', 'recall', 'f1'])
        fconv.close()

        start_epoch = 0
        print("=> no checkpoint found")

    # loop throuh epochs
    scaler = amp.GradScaler()
    for epoch in range(start_epoch, args.nepochs):
        start_time = time.time()
        # Train
        train_dset.shuffletraindata()
        train_whole_precision, train_whole_recall, train_whole_f1, train_whole_loss = train(epoch=epoch,
                                                                                            loader=train_loader,
                                                                                            model=model,
                                                                                            criterion=criterion,
                                                                                            optimizer=optimizer,
                                                                                            scaler=scaler,
                                                                                            batch_scheduler=scheduler,
                                                                                            accumulation=cfg["accumulation"])
        print('\tTraing Epoch: [{}/{}] Precision: {} Recall:{} F1score:{} Loss: {}'.format(epoch+1,
                  args.nepochs, train_whole_precision,train_whole_recall,train_whole_f1,train_whole_loss))
        result = [epoch+1, train_whole_precision, train_whole_recall, train_whole_f1, train_whole_loss]

        val_whole_precision, val_whole_recall, val_whole_f1, val_whole_loss, val_probs = val_predict(epoch=epoch,
                                                                                                     loader=val_loader,
                                                                                                     model=model,
                                                                                                     criterion=criterion)
        result.extend([val_whole_precision, val_whole_recall, val_whole_f1, val_whole_loss])

        msi_pro = group_proba(val_dset.slideIDX, val_probs, 0.5)
        roc_auc = roc_auc_score(val_dset.targets, msi_pro)
        best_roc_auc_threshold = get_roc_best_threshold(y_true=val_dset.targets,
                                                        y_score=msi_pro)
        mAP = average_precision_score(val_dset.targets, msi_pro)
        best_mAP_auc_threshold = get_mAP_best_threshold(y_true=val_dset.targets,
                                                        y_score=msi_pro)
        result.extend([round(roc_auc, 4), round(best_roc_auc_threshold, 5),
                       round(mAP, 4), round(best_mAP_auc_threshold, 5)])
        v_pred = np.where(msi_pro > 0.5, 1, 0)
        metrics_meters = calc_accuracy(v_pred, val_dset.targets)
        result.extend([metrics_meters['precision'], metrics_meters['recall'], metrics_meters['f1score']])
        str_logs = ['{} : {:.4}'.format(k, v) for k, v in metrics_meters.items()]
        s = ', '.join(str_logs)
        print('\tValidation  Epoch: [{}/{}] AUC: {} mAP:{} Loss: {} {}'.format(epoch+1,
              args.nepochs, roc_auc, mAP, val_whole_loss, s))

        fconv = open(os.path.join(args.output, time_mark + '_metrics.csv'), 'a',
                     encoding='utf-8-sig', newline='')
        writer = csv.writer(fconv)
        writer.writerow(result)
        fconv.close()

        eopch_save = {"state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(),
                      "epoch": epoch+1,
                      "dataset_state_list": [train_dset.slidenames, train_dset.targets,
                                             train_dset.batch, train_dset.grid]}
        torch.save(eopch_save, os.path.join(args.output,
                                            'checkpoint',
                                            time_mark + "_{}".format(epoch + 1) + '_weight.pth'))
        print('\tEpoch %d has been finished, needed %.2f sec.' % (epoch+1, time.time() - start_time))
    

def inference(run, loader, model, phase):
    model.eval()
    probs = np.zeros((1, 2), np.float32)
    whole_probably = 0.

    with torch.no_grad():
        with tqdm(loader, desc='Epoch:' + str(run+1) + ' ' + phase + '\'s inferencing',
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


def val_predict(epoch, loader, model, criterion):
    model.eval()
    probs = np.zeros((1, 2), np.float32)
    whole_loss = 0.
    whole_target = np.zeros((1, 1), np.int8)
    whole_predict = np.zeros((1, 1), np.int8)

    with tqdm(loader, desc='Epoch:' + str(epoch + 1) + ' is Val',
              file=sys.stdout, disable=False) as iterator:
        with torch.no_grad():
            for i, (input, target) in enumerate(iterator):
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                _, pred = torch.max(output.detach().cpu().to(torch.float32), 1)
                pred = np.int8(pred.data.numpy())
                target = target.cpu().numpy()
                whole_target = np.row_stack((whole_target, np.int8(target.reshape(-1, 1))))
                whole_predict = np.row_stack((whole_predict, pred.reshape(-1, 1)))

                output = F.softmax(output, dim=1)
                prob = output.detach().clone()
                prob = prob.cpu().numpy()
                probs = np.row_stack((probs, prob))
                batch_proba = np.mean(prob, axis=0)
                s = 'batch proba :' + str(batch_proba)
                iterator.set_postfix_str(s)
                whole_loss += loss.item()

    whole_target = np.delete(whole_target, 0, axis=0)
    whole_predict = np.delete(whole_predict, 0, axis=0)
    whole_metrics_meters = calc_accuracy(whole_predict, whole_target)

    probs = np.delete(probs, 0, axis=0)
    return whole_metrics_meters['precision'], whole_metrics_meters['recall'], whole_metrics_meters[
        'f1score'], round(whole_loss / (i + 1), 3), probs.reshape(-1, 2)


def train(epoch, loader, model, criterion, optimizer, scaler,
          batch_scheduler=None, accumulation=1):
    model.train()
    logs = {}
    whole_loss = 0.
    whole_target = np.zeros((1, 1) ,np.int8)
    whole_predict = np.zeros((1, 1), np.int8)

    iters = len(loader)
    with tqdm(loader, desc='Epoch:' + str(epoch+1) + ' is Train',
              file=sys.stdout, disable=False) as iterator:
        for i, (input, target) in enumerate(iterator):
            input = input.cuda()
            target = target.cuda()
            with amp.autocast():
                output = model(input)
                loss = criterion(output, target)
            _, pred = torch.max(output.detach().cpu().to(torch.float32), 1)
            pred = np.int8(pred.data.numpy())
            target = target.cpu().numpy()
            whole_target = np.row_stack((whole_target, np.int8(target.reshape(-1, 1))))
            whole_predict = np.row_stack((whole_predict, pred.reshape(-1, 1)))

            # 反向传播以及梯度更新
            scaler.scale(loss).backward()
            if (i + iters * epoch) % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if batch_scheduler is not None:
                    batch_scheduler.step(epoch+i*1.0/iters)

            metrics_meters = calc_accuracy(pred, target)
            logs.update(metrics_meters)
            logs.update({'loss':loss.item()})
            str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
            s = ', '.join(str_logs)

            iterator.set_postfix_str(s)
            whole_loss += loss.item()
            
    whole_target = np.delete(whole_target, 0, axis=0)
    whole_predict = np.delete(whole_predict, 0, axis=0)
    whole_metrics_meters = calc_accuracy(whole_predict, whole_target)

    return whole_metrics_meters['precision'],whole_metrics_meters['recall'],whole_metrics_meters['f1score'],round(whole_loss/(i+1),3)


if __name__ == '__main__':
    from config import native_cfg as cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
    parser.add_argument('--train_dir', type=str,
                        default='/Your_dir')
    parser.add_argument('--select_lib', type=str,
                        default=None,
                        help='path to validation MIL library binary. If present.')
    parser.add_argument('--train_lib', type=str,
                        default="/Your_dir/batch_train_sample.db",
                        help='path to train MIL library binary')
    parser.add_argument('--val_lib', type=str,
                        default="/Your_dir/batch_val_sample.db",
                        help='path to train MIL library binary')
    parser.add_argument('--output', type=str,
                        default='/Your_output_dir',
                        help='name of output file')
    parser.add_argument('--resume', type=bool, default=False, help='is the model retrained from the breakpoint')
    parser.add_argument('--resume_molde_path', type=str,
                        default=None,
                        help='The default is none, which means retraining from the last round. You can also specify the power down file path for the model, indicating which round to start from')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (default: 512)')
    parser.add_argument('--nepochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')

    args = parser.parse_args()
    main(args, cfg)
