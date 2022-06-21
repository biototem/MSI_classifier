import configparser
import os
import sys
import glob
sys.path.append('../')
import numpy as np
import cv2
import time
import pickle

from utils.openslide_utils import Slide
from utils.tissue_utils import Tissue, judge_not_white_proportion
# from CUT.stain_trans import standard_transfrom
from utils.file_utils import getAllImageName, getAllImagePath
from utils.DataReader import load_data


def svs_cut(svs_path, level, img_save_dir, patch_size, step=None,
            binary_label=None, resize=None, tissue_cfg=None,
            is_normalization=False, normalization_cfg=None):
    if step is None:
        step = patch_size
    svs_name = (os.path.basename(svs_path)).strip(".svs")
    slide = Slide(svs_file=svs_path)

    # if is_normalization and normalization_cfg is not None:
    #     sttd = cv2.imread(normalization_cfg["sttd_patch_path"])
    #     sttd = cv2.cvtColor(sttd, cv2.COLOR_BGR2RGB)
    #     stain_method = standard_transfrom(sttd, normalization_cfg["normalization_method"])

    # 获取相应级数下采样的图片尺寸
    svs_width, svs_height = slide.get_level_dimension(level=level)
    svs_downsample = int(slide.get_level_downsample(level=level))

    tissue = Tissue(slide, binary_label, level=2,
                    binary_threshold=tissue_cfg["binary_threshold"],
                    contour_area_threshold=tissue_cfg["contour_area_threshold"])
    grid = []
    for j in range(0, svs_height, step):
        for k in range(0, svs_width, step):
            # 跳过超出边界之外的patch
            if k + patch_size >= svs_width or j + patch_size >= svs_height:
                continue

            if tissue.judeg_tissue_proportion((k, j), level, (patch_size, patch_size),
                                              tissue_threshold=tissue_cfg["tissue_threshold"]):
                slide_region = np.array(
                    slide.read_region((k * svs_downsample, j * svs_downsample),
                                      level, (patch_size, patch_size)))[:, :, 0:3]

                try:
                    # 加入这个判断是为了保证对可能出现的纯白色图片不做处理,直接丢弃
                    if judge_not_white_proportion(slide_region, not_white_threshold=tissue_cfg["not_white_threshold"]):
                        # if is_normalization:
                        #     slide_region = stain_method.transform(slide_region)

                        save_img_name = f"{svs_name}_{k}_{j}.jpg"
                        if os.path.exists(os.path.join(img_save_dir, save_img_name)):
                            grid.append((k, j))
                            continue

                        # print(os.path.join(img_save_dir,svs_name + '_' + str(k) + '_' + str(j) + '.jpg'))
                        if resize is not None and isinstance(resize, int) and resize > 0:
                            slide_region = cv2.resize(slide_region, (resize, resize))
                        cv2.imwrite(os.path.join(img_save_dir, save_img_name),
                                    cv2.cvtColor(slide_region, cv2.COLOR_RGB2BGR))
                        grid.append((k, j))

                    else:
                        continue

                except:
                    if is_normalization:
                        cv2.imwrite(os.path.join(img_save_dir, save_img_name),
                                    cv2.cvtColor(slide_region, cv2.COLOR_RGB2BGR))
                    else:
                        pass
    slide.close()
    return grid


def batch_cut_img(cfg, classes_to_label, work_name_list=None):
    # svss = glob.glob(os.path.join(svs_dir, "*.svs"))
    svss = getAllImagePath(cfg["svs_dir"])
    svss = sorted(svss)
    if not os.path.exists(cfg["save_dir"]):
        os.makedirs(cfg["save_dir"])
    db_path = os.path.join(os.path.dirname(cfg["save_dir"]),
                           os.path.basename(cfg["save_dir"])+".db")

    if os.path.exists(db_path):
        with open(db_path, "rb") as fp:
            lib = pickle.load(fp)
    else:
        lib = {"slides": [], "grid": [], "level": cfg["level_count"], "batch": [],
               "patch_size": [], "targets": []}
    nomask = []
    # 提取mask对应的轮廓
    tissue_cfg = {}
    tissue_cfg["binary_threshold"] = cfg["binary_threshold"]
    tissue_cfg["contour_area_threshold"] = cfg["contour_area_threshold"]
    tissue_cfg["tissue_threshold"] = cfg["tissue_threshold"]
    tissue_cfg["not_white_threshold"] = cfg["not_white_threshold"]

    # 颜色标准化参数
    is_normalization = cfg["is_normalization"]
    normalization_cfg = {}
    normalization_cfg["sttd_patch_path"] = cfg["sttd_patch_path"]
    normalization_cfg["normalization_method"] = cfg["normalization_method"]

    for svs in svss:
        # 判断是否存在标签
        svs_name = os.path.splitext(os.path.basename(svs))[0]
        if classes_to_label.get(svs_name, None) is None\
                or svs_name in lib["slides"]:
            continue
        start_time = time.time()
        print("Starting cut SVS image %s..." % svs)

        if work_name_list is None or len(work_name_list) == 0:
            pass
        else:
            if svs_name not in work_name_list:
                continue

        # 加载SVS图片
        img_save_dir = os.path.join(cfg["save_dir"], svs_name)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        # 加载对应掩码
        try:
            if cfg["mask_dir"] is None or cfg["mask_dir"] == 0 or cfg["mask_dir"] is False:
                binary_label = None
            elif isinstance(cfg["mask_dir"], str) and os.path.exists(cfg["mask_dir"]):
                binary_label = os.path.join(cfg["mask_dir"], svs_name+".png")
            elif isinstance(cfg["mask_dir"], dict):
                binary_label = cfg["mask_dir"][svs_name]
        except:
            pass

        grid = svs_cut(svs, cfg["level_count"], img_save_dir, cfg["patch_size"],
                       cfg["step"], binary_label, cfg["resize"], tissue_cfg,
                       is_normalization, normalization_cfg)

        if len(grid) == 0:
            nomask.append(svs_name)
            continue
        lib["slides"].append(svs_name)
        lib["grid"].append(grid)
        lib["batch"].append(os.path.basename(cfg["save_dir"]))
        lib["patch_size"].append(cfg["patch_size"])
        lib["targets"].append(classes_to_label[svs_name])
        with open(db_path, "wb") as fp:
            pickle.dump(lib, fp)

        print("Finished cut SVS image %s, needed %.2f sec" % (svs, time.time() - start_time))

    print(nomask)


if __name__ == "__main__":
    from DataProcessing.process_confing import cut_cfg as cfg
    import pandas as pd
    work_name_list = None

    # 加载标签
    file_name = sorted([i.split(".")[0] for i in os.listdir(
        "/media/biototem/Elements/Colon_MSI_batch_5_CUT/Mask/nine_mask_visual/batch_3_SYSUCC")])
    label_file = "/media/totem_disk/totem/haosen/target_new.xlsx"
    df = pd.read_excel(label_file, sheet_name="batch_3")
    classes_to_label = {}
    table_name = [str(i).strip(".svs") for i in df["file_name"]]
    label = list(df["class"])
    for index, name in enumerate(table_name):
        if name in file_name:
            classes_to_label[name] = label[index]
        elif name+"_" in file_name:
            classes_to_label[name+"_"] = label[index]

    # file_name = sorted([i.split(".")[0] for i in os.listdir(
    #     "/media/biototem/Elements/Colon_MSI_batch_5_CUT/Mask/nine_mask_visual/batch_3_SYSUCC")])
    # label_file = "/media/totem_disk/totem/haosen/target_new.xlsx"
    # df = pd.read_excel(label_file, sheet_name="batch_5")
    # classes_to_label = {}
    # table_name = [str(i).strip(".svs") for i in df["file_name"]]
    # label = list(df["class"])
    # for index, name in enumerate(table_name):
    #     if name in file_name:
    #         classes_to_label[name] = label[index]
    #     elif name+"_" in file_name:
    #         classes_to_label[name+"_"] = label[index]
    #
    # img_name = getAllImageName("/media/biototem/Elements/Colon_MSI_batch_5")
    # classes_to_id = dict(zip(img_name, [0]*len(img_name)))
    # classes_to_label.update(classes_to_id)
    #
    # path = ["/media/biototem/Elements/Colon_MSI_batch_5_CUT/Mask/zunhu_mask/batch_3_SYSUCC_mask",
    #         "/media/biototem/Elements/Colon_MSI_batch_5_CUT/Mask/zunhu_mask/batch_5_SYSUCC_mask"]
    # mask_path = glob.glob(os.path.join(path[0], "*")) \
    #             + glob.glob(os.path.join(path[1], "*"))
    # mask_dir = dict(zip([os.path.splitext(os.path.basename(i))[0] for i in mask_path], mask_path))
    # cfg["mask_dir"] = mask_dir
    batch_cut_img(cfg, classes_to_label, work_name_list=work_name_list)
