# 100,000 histological images of human colorectal cancer and healthy tissue

**Data Description "NCT-CRC-HE-100K"**

- This is a set of 100,000 non-overlapping image patches from hematoxylin & eosin (H&E) stained histological images of human colorectal cancer (CRC) and normal tissue.
- All images are 224x224 pixels (px) at 0.5 microns per pixel (MPP). All images are color-normalized using Macenko's method (http://ieeexplore.ieee.org/abstract/document/5193250/, DOI [10.1109/ISBI.2009.5193250](https://doi.org/10.1109/ISBI.2009.5193250)).
- Tissue classes are: Adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), colorectal adenocarcinoma epithelium (TUM).
- These images were manually extracted from N=86 H&E stained human cancer tissue slides from formalin-fixed paraffin-embedded (FFPE) samples from the NCT Biobank (National Center for Tumor Diseases, Heidelberg, Germany) and the UMM pathology archive (University Medical Center Mannheim, Mannheim, Germany). Tissue samples contained CRC primary tumor slides and tumor tissue from CRC liver metastases; normal tissue classes were augmented with non-tumorous regions from gastrectomy specimen to increase variability.

**Ethics statement "NCT-CRC-HE-100K"**

All experiments were conducted in accordance with the Declaration of Helsinki, the International Ethical Guidelines for Biomedical Research Involving Human Subjects (CIOMS), the Belmont Report and the U.S. Common Rule. Anonymized archival tissue samples were retrieved from the tissue bank of the National Center for Tumor diseases (NCT, Heidelberg, Germany) in accordance with the regulations of the tissue bank and the approval of the ethics committee of Heidelberg University (tissue bank decision numbers 2152 and 2154, granted to Niels Halama and Jakob Nikolas Kather; informed consent was obtained from all patients as part of the NCT tissue bank protocol, ethics board approval S-207/2005, renewed on 20 Dec 2017). Another set of tissue samples was provided by the pathology archive at UMM (University Medical Center Mannheim, Heidelberg University, Mannheim, Germany) after approval by the institutional ethics board (Ethics Board II at University Medical Center Mannheim, decision number 2017-806R-MA, granted to Alexander Marx and waiving the need for informed consent for this retrospective and fully anonymized analysis of archival samples).

├── NCT-CRC-HE-100K-NORM
│   ├── ADI
│   ├── BACK
│   ├── DEB
│   ├── LYM
│   ├── MUC
│   ├── MUS
│   ├── NORM
│   ├── STR
│   └── TUM

**Data set "CRC-VAL-HE-7K"**

This is a set of 7180 image patches from N=50 patients with colorectal adenocarcinoma (no overlap with patients in NCT-CRC-HE-100K). It can be used as a validation set for models trained on the larger data set. Like in the larger data set, images are 224x224 px at 0.5 MPP. All tissue samples were provided by the NCT tissue bank, see above for further details and ethics statement.

├── CRC-VAL-HE-7K
│   ├── ADI
│   ├── BACK
│   ├── DEB
│   ├── LYM
│   ├── MUC
│   ├── MUS
│   ├── NORM
│   ├── STR
│   └── TUM

**Data set "NCT-CRC-HE-100K-NONORM"**

This is a slightly different version of the "NCT-CRC-HE-100K" image set: This set contains 100,000 images in 9 tissue classes at 0.5 MPP and was created from the same raw data as "NCT-CRC-HE-100K". However, no color normalization was applied to these images. Consequently, staining intensity and color slightly varies between the images. Please note that although this image set was created from the same data as "NCT-CRC-HE-100K", the image regions are not completely identical because the selection of non-overlapping tiles from raw images was a stochastic process.

**General comments**

Please note that the classes are only roughly balanced. Classifiers should never be evaluated based on accuracy in the full set alone. Also, if a high risk of training bias is excepted, balancing the number of cases per class is recommended.

├── NCT-CRC-HE-100K-NONORM
│   ├── ADI
│   ├── BACK
│   ├── DEB
│   ├── LYM
│   ├── MUC
│   ├── MUS
│   ├── NORM
│   ├── STR
│   └── TUM
