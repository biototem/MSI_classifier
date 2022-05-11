# MSI classifier

This repository provides source code for the article *Clinical Actionability of Triaging DNA Mismatch Repair Deficient Colorectal Cancer from Biopsy Samples using Deep Learning*.

# Installtation

The running environment of this repository base on  python3.8, CUDA >=11.2 and pytorch >=1.6

## Requirements

```bash
pip install -r requirements.txt
```

The code of `MIL_method` needs timm python package ,which can refer to the author's github: [timm library](https://github.com/rwightman/pytorch-image-models)

If `pip` installation of the timm package fails, you can search for timm on pypi, then download the whl file to the local, and  install it locally.

# CODE DESCRIPTION

## base_model_for_CRC_9_classification

This directory provides training and testing scripts of colorectal cancer prediction (base on CRC 9 CLASS NCT-CRC-HE-100K-NONORM dataset).  You can see more details in **./base_model_for_CRC_9_classification/README.md**.

## MIL_method

This directory provides training and testing scripts of MIL & naive method  for microsatellite instability predicting in CRC histology. You can see more details in **./MIL_method/README.md**.
