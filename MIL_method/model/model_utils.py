import torch
import torch.nn as nn
import timm
from torchsummary import summary


def get_pretrained_model(model_name, model_weight_path=None, num_classes=0, save_classifier=True,
                         img_width=224, img_height=224, pretrained=True, parallel=False, verbose=True):
    """
    model_name: 模型名字
    model_weight_path：模型权重路径，如果为None，则表示不加载任何权重，同时pretrained无效
    				   如果不为None，如果pretrained为True，表示使用的预训练权重，为False的话，则表示是自己训练的权重
    num_classes：类别数量
    save_classifier: 是否只保留分类层，如果为False则用于特征提取
    img_height：输入图片的高
    img_width：输入图片的宽
    pretrained：是否是预训练权重
    parallel：是否用nn.DataParallel包装模型
    verbose：是否打印模型结构
    """
    if model_weight_path is None:
        model = timm.create_model(model_name=model_name,
                                  pretrained=pretrained,
                                  num_classes=num_classes)
        if parallel:
            parallel_model = nn.DataParallel(model)
        else:
            parallel_model = model
    else:
        if num_classes == 0:
            raise ValueError("the num_classes not zero, ")
        if pretrained:
            model_num_classes = 1000
        else:
            model_num_classes = num_classes
        model = timm.create_model(model_name=model_name,
                                  pretrained=False,
                                  num_classes=model_num_classes)
        if parallel:
            parallel_model = nn.DataParallel(model)
        else:
            parallel_model = model
        state_dict = torch.load(model_weight_path)
        parallel_model.load_state_dict(state_dict
                                       if state_dict.get("state_dict", None) is None
                                       else state_dict["state_dict"])
        if pretrained:
            n_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features=n_features, out_features=num_classes)

    if num_classes != 0 and not save_classifier:
        model.reset_classifier(0)

    if verbose is True:
        model.cpu().eval()
        with torch.no_grad():
            summary(model, input_size=(3, img_height, img_width), device="cpu")
    return parallel_model



if __name__ == "__main__":
    get_pretrained_model(model_name="densenet121",
                         model_weight_path="/root/.cache/torch/checkpoints/densenet121_ra-50efcf5c.pth",
                         num_classes=1000,
                         save_classifier=False)