cut_cfg={
    # file path of WSI(svs format)
    "svs_dir": ["your svs file path"],
    # mask，if None，means using tissue region in processing;otherwise using a binary label image(png format) in processing and the image file path must be Ture
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
    "patch_size": 448,  # size of getting tiles form WSI in openslide read_region method
    "step": 448,    # step length in etting tiles
    "resize": 224,   # pixel size of saving sample image

    # color normalization or not，which is not used anymore
    "is_normalization": False,
    "sttd_patch_path": None,
    "normalization_method": None,
}

