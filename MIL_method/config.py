native_cfg={
    "img_width": 224,  # width of input images
    "img_height": 224, # height of input images
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
}


