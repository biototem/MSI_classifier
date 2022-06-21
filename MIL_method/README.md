# Multiple-Instance learning for microsatellite instability predicting in CRC histology

This directory provides MIL & naive training and tesing scripts.

Naive method just training  a typical MSI/MSS tile classifier base on  the naive training data，which MSI iamges all sampled from MSI histology and MSS iamges all sampled from MSS histology

**Workflow**：

1. Generate training data
2. MIL/Naive Training
3. MIL/Naive Testing

## Specific process

1. Edit **DataProcessing/process_confing.py** and modify relevant parameters.You can run **cut.py** to generate training data and testing data(images form tiles in WSI).

2. you can run **DatasetSplit.py** to divide  whole  central dataset into training set and validation set. 

3. When trainning data is available, run **MIL_train.py**  or **naive_train.py** for training.

4. Call relevant predicttion scripts for evaluation.

## Scripts description

There are three subdirectories and three python scripts under the project,here is the description of these scripts.

##### 1. DataProcessing

This subdirectory  provides scripts for  data processing.

##### (1) process_confing.py

```python
cut_cfg={
    # file path of WSI(svs format) 
    "svs_dir": ["your WSIs file path"],
    # mask，if None，means using tissue region in processing，
    # otherwise using a binary label image(png format) in processing and the image file path must be Ture
    "mask_dir": "your masks file path",
    # sampling results save file path
    "save_dir": "sampling results save file path",   

    # when mask is None,the parameters below is used for getting tissue regions from WSI
    "binary_threshold": 230,    # threshold for WSI thumbnail to binary image
    "contour_area_threshold": 0.0001,   # the minimum proportion of contour in whole thumbnail
    # parameters below is used for saving sample tiles to images
    "tissue_threshold": 0.9,    # proportion of tumor region area and whole region area
    "not_white_threshold": 0.5,     # white pixel proportion threshold

    "level_count": 0,   # process level in openslide
    "patch_size": 512,  # size of getting tiles form WSI in openslide read_region method
    "step": 512,    # step length in etting tiles
    "resize": 224,   # pixel size of saving sample image
}
```

##### (2) cut.py

This script is for getting  tile patches from WSIs (saved as image format), and stored the corresponding  information into a python dictionary,  which saved as a binary python pickle file and would be loaded by the dataloader in the subsequent training process. In addition to the generated training sample, the format of the saved file is described below. Here are meanings of the keys of the dictionary:

- slides： whole file name of WSI

- grid:：a list of tuple (x,y) upper left coordinates of each WSI

- level：WSI pyramid level (integer) from which to read the tiles. Usually `0` for the highest resolution.

- batch：the file source name(usually is the outermost folder name) of WSI to process

- patch_size：patch size

- targets：label list of all WSI (0: MSS, 1: MSI). Size of the list equals the number of slides.

The  example targets file is available at: **./DataProcessing/TCGA-CRC_target.csv**

##### (3) DatasetSplit.py

Script for dividing  training dataset and validation dataset which according to the proportion of original WSI file, but it  is not necessary in the  single data center situation.

##### 2. config.py

Define the training parameters：

```python
    "img_width": 224,    # width of input images
    "img_height": 224,    # height of input images
    "num_classes": 2,   # 2 for binary classification
    "lr": 1e-4,     # learning rate
    "min_lr": 1e-4,     # minimum learning rate
    "weight_decay": 1e-5,   # weight decay
    "accumulation": 1,    # accumulation gradient steps

    # parameters for ETL process
    # 'grid_ mode' and 'grid_ drop'is a setting parameter used to discard too many patches in a slide
    # there are two modes in grid_mode, no_ drop-- do not discard; fixed_ drop-- fixed discard grid_ patch of drop scale
    "grid_mode": "no_drop",
    "grid_drop": 0.95,     # discard ratio
    "drop_limit": 300,     # minimum for discarding
    "data_balance": False,  # data balance or not
```

Since the training process uses CosineAnnealingWarmRestarts as the adjustment learning rate, if you don't want to use it, you can set **lr** equals **min_lr**, that the learning rate would be constant. Or set the batch_scheduler in the train function to `None`. It should also be noted that this project updates the learning rate after each batch, not after each epoch. If you want to use other schedules, you need to modify the code yourself, and you need to determine whether to update the learning rate after the epoch or after the end of the batch.

##### 3. naive_train.py & MIL_train.py

The training function can be run directly in the terminal:

```bash
python naive_train.py
```

```bash
python MIL_train.py
```

The following explains the various parameter functions for use in the terminal

```python
'--train_dir'： 'directory(whole file path) of input data'
'--select_lib'：'path to validation MIL library binary. If present.'
'--train_lib'： 'path to train dataset library binary generated by data processing'
'--val_lib'：'path to val dataset library binary generated by data processing'
'--output'：'base directory for output files '
'--resume'：'The default is none, which means retraining from the last round. You can also specify the power down file path for the model, indicating which round to start from'
'--resume_molde_path'： 'The default is none, which means retraining from the last round. You can also specify the power down file path for the model, indicating which round to start from'
'--batch_size'： 'mini-batch size (default: 128)'
'--nepochs'：'number of epochs'
'--workers'：'number of data loading workers (default: 0)'
'--weight'：'unbalanced positive class weight (default: 0.5, balanced classes)'
'--k'：'top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)'
```

It is not recommended to use the parameter form to run in the terminal, which is more troublesome. Instead, define the parameters in the script and run it directly in the IDE or in the terminal.

##### 4. NaivePredict.py

The model evaluation function, which is compatible with the evaluation of naive and MIL training methods, can be entered directly in the terminal:

```bash
python NaivePredict.py
```

The following explains the various parameter functions for use in the terminal

```python
'--data_lib'：'path to whole predeict data library binary.'
'--select_lib'：'path to validation MIL library binary. If present.'
'--data_dir'：'directory(whole file path) of input data'
'--output_dir'：'base directory for output files '
'--batch_size'：'mini-batch size (default: 128)'
'--img_width'：'width of input images'
'--img_hight'：'height of input images'
'--model_weights'：'model weights file path' 
'--threshold'：'threshold of classification judgment,default 0.5'
'--k'：'top k tiles in MIL training ,in prediction of naive training method,it should be set to 0.'
'--workers'：'number of data loading workers (default: 0)'
```

##### 4. MILPredict function

The usage example of predict function is available in **MIL_train.py**. When you define you predict loader,you can call the prediction function as follows:

```python
from utils.DataReader import MILdataset
from utils.train_utils import group_proba, calc_accuracy, group_argtopk, group_identify
from utils.nn_metrics import get_roc_best_threshold, get_mAP_best_threshold
from config import native_cfg as cfg
import albumentations
import torch

val_trans = albumentations.Compose([albumentations.Resize(cfg["img_hight"], cfg["img_width"]),
                                        ToTensorV2()])

val_dset = MILdataset(args.val_lib, args.k, val_trans,args.train_dir,
                          select_condition=target_val_slide)
val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

val_dset.setmode(1)
val_whole_precision,val_whole_recall,val_whole_f1,val_whole_loss,val_probs = train_predict(epoch, val_loader, model, criterion, optimizer, 'val')
msi_pro = group_proba(val_dset.slideIDX, val_probs, 0.5)
roc_auc = roc_auc_score(val_dset.targets, msi_pro)
best_roc_auc_threshold = get_roc_best_threshold(y_true=val_dset.targets,
                                                        y_score=msi_pro)
mAP = average_precision_score(val_dset.targets, msi_pro)
best_mAP_auc_threshold = get_mAP_best_threshold(y_true=val_dset.targets,
                                                        y_score=msi_pro)
```

##### 6. model

This directory encapsulates the model structure of two densenet_ibn and the call method of the model library `timm`. The calling method of densenet_ibn as follows:

```python
# num_ classes is the number of categories, and pretrained can only be false. Because Ibn structure is not in original densenet, there is no corresponding pre training weight
model = densenet121_ibn_b(num_classes, pretrained=False)
```

model_utils.py encapsulates the call of the model library timm, mainly to call the efficientnet_ns series models. The specific usage is as follows:

```python
"""
model_name: name of model 
model_weight_path：model weight file path.If None，means no wetght to be loaded，and pretrained mode is falsed;
                   if not None，and pretrained is True，means loads pretrained model weight,when set to False,
                   means loads the weight you trained.
num_classes：the number of categories
save_classifier: Keep the last sofmax layer for classification or not，if False means used to feature extraction.
img_height：width of input images
img_width：height of input images
pretrained：是否是预训练权重
parallel：是否用nn.DataParallel包装模型
verbose：是否打印模型结构
"""
model = get_pretrained_model(model_name="tf_efficientnet_b7_ns",
                             model_weight_path=None,
                             num_classes=2,
                             img_height=cfg["img_hight"],
                             img_width=cfg["img_width"],
                             pretrained=False,
                             parallel=False, 
                             verbose=True)
```

##### 7. utlis

It mainly encapsulates some auxiliary functions, which are used for data reading in the training process. Mainly, DataReader.py encapsulates the data loading class, which is compatible with the data reading methods of naive and MIL.
