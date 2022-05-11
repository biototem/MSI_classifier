# DenseNet with Instance-Batch Normalization for image classification

This directory provides training and testing scripts for classification of hematoxylin & eosin (H&E) stained histological images of human colorectal cancer (CRC) and normal tissue

## Data Description

**./DATA** is the subdirectory to store CRC 9 class dataset 's data file(image format) for training and testing scripts . You can see more details in **./DATA/README.md**.

## ETL

**./ETL** is the subdirectory of ETL processing scripts.

**./ETL/fold_spilt.py**

This script is used to generate the three-fold data list(python TYPE LIST) of the dataset with a random method(or any other method that writed by yourself), and saved as a binary python pickle file. 

**./ETL/paper_sample_show.py**

This script is used to get a overview picture of the dataset you choosed base on every classes.

## Train

**./train_and_test** include training and testing scripts.

1. Edit `train.py`. Modify `model` and `data_path` and other **parameter** to yours.

```python
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
```

2. Run train script
   
   ```bash
   cd ./train_and_test
   python train.py
   ```

## Test

**./train_and_test/test.py** is the testing scripts.

In our pytorch datalaoder method, it should define the datasource (dataset file path or data list(TYPE LIST,loaded by the  binary python pickle file, which helps to  load the test dataset into dataloder that you want to predict and static.

1. You can define three-fold dataset like bellow:

```python
    with open('../ETL/fold1_train_list.db','rb') as FP:
        fold1_list = pickle.load(FP)
    with open('../ETL/fold2_train_list.db','rb') as FP:
        fold2_list = pickle.load(FP)
    with open('../ETL/fold3_train_list.db','rb') as FP:
        fold3_list = pickle.load(FP)
    val_list = fold3_list
    train_list = fold1_list + fold2_list
    fold_mark =  'fold_3'
    test_dset = augmention_dataset(sub_dir = None,
                                   class_to_idx = None, 
                                   image_list = train_list,# or val_list
                                   transform=val_trans)
```

Or you can make the original data directory as a whole dataset to test, by setting sub_dir with a true path and image_list is None  :

```python
  test_dset = augmention_dataset(sub_dir = '../DATA/NCT-CRC-HE-100K-NONORM',
                                   class_to_idx = None, 
                                   image_list = None, 
                                   transform=val_trans)
```

2. Run train script
   
   ```bash
   cd ./train_and_test
   python test.py
   ```
