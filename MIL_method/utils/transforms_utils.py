import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def transforms_visual(img_path, trans=None, img_numbers=9):
    plt.figure()
    col = 5
    row = (img_numbers // col) + 1 if img_numbers % col > 0 else (img_numbers // col)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(row, col, 1)
    plt.imshow(np.array(img))
    plt.title("origin")
    plt.axis("off")
    for i in range(img_numbers):
        img_new = trans(image=img)["image"]
        plt.subplot(row, col, i+2)
        plt.imshow(img_new)
        plt.title(str(i+1))
        plt.axis("off")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import albumentations

    aug = albumentations.Compose([
        albumentations.VerticalFlip(),
        albumentations.HorizontalFlip(),
        albumentations.Rotate(360),
        albumentations.ShiftScaleRotate(),
        albumentations.HueSaturationValue(hue_shift_limit=3),
        albumentations.GaussianBlur(blur_limit=(1, 3)),
        albumentations.RandomScale(scale_limit=0.2),
        albumentations.Resize(224, 224)
    ])

    root = "/media/biototem/Elements/Colon_MSI_batch_5_CUT/20x_patch=512_step=512/batch_1_SYSUCC/18660_616043001"
    img_dir = [os.path.join(root, i) for i in os.listdir(root)]
    img_path = img_dir[150]
    transforms_visual(img_path, aug, 14)