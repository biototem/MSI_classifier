import cv2
import numpy as np
from utils.openslide_utils import Slide
from utils.opencv_utils import OpenCV
from utils.xml_utils import xml_to_region, region_binary_image
from PIL import Image


class Tissue(object):
    def __init__(self, slide, binary_label,level=2,
                 **kwargs):
        self.slide = slide
        self.level = level if level <= len(self.slide.level_dimensions)-1 else len(self.slide.level_dimensions)-1
        self.level_downsample = int(self.slide.get_level_downsample(level=self.level))
        self.tissue = self.get_tissue_matrix(binary_label, cfg=kwargs)
        # img = np.array(self.slide.get_thumb(level=self.level).convert("RGB"))[:, :, :3]
        # cv2.imshow("1", cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (800, 600)))
        # cv2.imshow("", cv2.resize(self.tissue,(800, 600))*255)
        # cv2.waitKey()

    def get_tissue_matrix(self, path, cfg={}):
        if path is None:
            binary_threshold = cfg.get("binary_threshold", None)
            contour_area_threshold = cfg.get("contour_area_threshold", None)
            mode = cfg.get("mode", 0)
            erode_iter = cfg.get("erode_iter", 0)
            dilate_iter = cfg.get("dilate_iter", 0)
            if binary_threshold is None or contour_area_threshold is None:
                raise ValueError("binary_threshold and contour_area_threshold is not None")

            mask = get_infill_tissue_matrix(np.array(self.slide.get_thumb(level=self.level))[:, :, 0:3],
                                            binary_threshold=binary_threshold,
                                            contour_area_threshold=contour_area_threshold,
                                            mode=mode, erode_iter=erode_iter, dilate_iter=dilate_iter)

        elif path.endswith(".tif"):
            im = cv2.imread(path)
            mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if np.max(mask) == 1:
                mask = mask.astype(np.uint8)
            else:
                mask = (mask / np.max(mask)).astype(np.uint8)

        elif path.endswith(".xml"):
            tile = self.slide.get_thumb(level=self.level)
            region_list, region_class = xml_to_region(path)
            mask = region_binary_image(tile, region_list, region_class, self.level_downsample)

        elif path.endswith(".npy"):
            mask = np.load(path)
            mask[mask < 8] = 0
            mask[mask == 8] = 1

        elif path.endswith(".png"):
            im = cv2.imread(path)
            mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mask[mask < 128] = 0
            mask[mask >= 128] = 1

        # 判断是否符合类型和大小
        if not isinstance(mask, np.uint8):
            mask = mask.astype(np.uint8)
        if self.slide.get_level_dimension(self.level) != (mask.shape[1], mask.shape[0]):
            mask = cv2.resize(mask, self.slide.get_level_dimension(self.level))

        return mask

    def judeg_tissue_proportion(self, location, level_location, size, tissue_threshold):
        location_level_downsample = int(self.slide.get_level_downsample(level=level_location))
        scale = location_level_downsample / self.level_downsample

        location = [int(i*scale) for i in location]
        if isinstance(size, int):
            size = [int(size*scale), int(size*scale)]
        elif isinstance(size, tuple):
            size = [int(i*scale) for i in size]
        else:
            raise ValueError("size should be int, tuple or list")

        region = self.tissue[location[1]:location[1]+size[1], location[0]:location[0]+size[0]]
        # print(np.sum(region)/(region.shape[0]*region.shape[1]))
        # cv2.imshow("", region*255)
        # cv2.waitKey()
        if np.sum(region)/(region.shape[0]*region.shape[1]) > tissue_threshold:
            return True
        else:
            return False


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'.
    Specifically, a pixel is not white if its luminance (in LAB color space) is less than the specified threshold.
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)


def judge_not_white_proportion(im, not_white_threshold=0.01, threshold=0.8):
    mask = notwhite_mask(im, thresh=threshold).reshape((-1,))
    # print(np.sum(mask) / mask.size)
    if np.sum(mask) / mask.size > not_white_threshold:
        return True
    else:
        return False


def get_infill_tissue_matrix(im, binary_threshold=210, contour_area_threshold=0.001,
                             mode=0, erode_iter=0, dilate_iter=0):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    opencv = OpenCV(im)
    cnts = opencv.find_contours(is_erode_dilate=True, thresh=binary_threshold, mode=mode,
                                erode_iter=erode_iter, dilate_iter=dilate_iter)

    # 获取当前区域面积
    area_cnt_list = list(map(cv2.contourArea, cnts))
    # 获取当前区域占矩阵区域的面积比率
    rate_area = [area / im.size for area in area_cnt_list]
    # 过滤当前区域占矩阵区域的面积比率小于等于filter_rate的区域
    tissue_cnts = [cnts[i] for i in range(len(cnts)) if rate_area[i] >contour_area_threshold]

    # initialize mask to zero
    mask = np.zeros((im.shape[0], im.shape[1])).astype(im.dtype)
    color = [1]
    mask = cv2.fillPoly(mask, tissue_cnts, color)

    return mask