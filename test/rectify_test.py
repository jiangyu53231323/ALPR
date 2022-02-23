import math

import numpy
import numpy as np
import cv2
import copy
import random
import albumentations as A
import imgaug as ia
import torch
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from PIL import Image, ImageDraw, ImageFont


def resize_rectify(image, bbox, segmentation, size=(224, 64), is_rectify=True):
    '''
    :param image:
    :param bbox: [x1,y1,x2,y2]
    :param segmentation: [x1,y1,x2,y2,x3,y3,x4,y4]
    :param size: (w,h)
    :return:
    '''
    h, w = image.shape[0], image.shape[1]
    left = min(bbox[0], segmentation[0], segmentation[2])
    rigth = max(bbox[2], segmentation[4], segmentation[6])
    top = min(bbox[1], segmentation[1], segmentation[7])
    bottom = max(bbox[3], segmentation[3], segmentation[5])
    # 适当扩大10％的裁切范围，如果车牌区域太大则扩大固定范围
    padding_w = int((rigth - left) * 0.1) if ((rigth - left) * 0.1) < 44.8 else 44
    padding_h = int((bottom - top) * 0.1) if ((bottom - top) * 0.1) < 12.8 else 12
    # 防止越界
    left = int(left - padding_w) if (left - padding_w) > 0 else 0
    rigth = int(rigth + padding_w) if (rigth + padding_w) < (w - 1) else w - 1
    top = int(top - padding_h) if (top - padding_h) > 0 else 0
    bottom = int(bottom + padding_h) if (bottom + padding_h) < (h - 1) else h - 1
    h_new = bottom - top
    w_new = rigth - left

    image = image[top:bottom + 1, left:rigth + 1, :]
    # 防止越界
    keypoints = [(int(segmentation[0] - left) if (segmentation[0] - left) > 0 else 0,
                  int(segmentation[1] - top) if (segmentation[1] - top) > 0 else 0),
                 (int(segmentation[2] - left) if (segmentation[2] - left) > 0 else 0,
                  int(segmentation[3] - top) if (segmentation[3] - top) < (h_new - 1) else h_new - 1),
                 (int(segmentation[4] - left) if (segmentation[4] - left) < (w_new - 1) else w_new - 1,
                  int(segmentation[5] - top) if (segmentation[5] - top) < (h_new - 1) else h_new - 1),
                 (int(segmentation[6] - left) if (segmentation[6] - left) < (w_new - 1) else w_new - 1,
                  int(segmentation[7] - top) if (segmentation[7] - top) > 0 else 0), ]
    category_ids = [1]
    category_id_to_name = {1: 'PL'}
    transform = A.Compose([
        A.Resize(64, 224)],
        keypoint_params=A.KeypointParams(format='xy'))
    # try:
    transformed = transform(image=image, keypoints=keypoints, category_ids=category_ids)
    # except:
    #     print(img_path)
    #     print(segmentation)
    #     print(keypoints)
    image_new = transformed['image']
    keypoints_new = transformed['keypoints']

    if is_rectify == True:
        pts = np.float32([[keypoints_new[0][0], keypoints_new[0][1]], [keypoints_new[1][0], keypoints_new[1][1]],
                          [keypoints_new[2][0], keypoints_new[2][1]], [keypoints_new[3][0], keypoints_new[3][1]]])
        pts1 = np.float32(
            [[8, 3], [8, 62], [216, 62], [216, 3]])

        M = cv2.getPerspectiveTransform(pts, pts1)
        dst = cv2.warpPerspective(image_new, M, size)
        return dst
    else:
        return image_new


def get_lp(lp_number):
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                 "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    lp_str = ""
    for i in range(len(lp_number)):
        if i == 0:
            lp_str = lp_str + provinces[lp_number[i]]
        else:
            lp_str = lp_str + ads[lp_number[i]]
    return lp_str


def main():
    image = cv2.imread(
        "E:\\Project\\python_project\\ALPR\\img\\0058-8_20-275&491_361&548-361&548_285&537_275&491_351&502-0_0_20_1_24_31_29-145-72.jpg")  # [:, :, ::-1]
    cv2.imshow("image", image)

    lp_number = [0, 0, 20, 1, 24, 31, 29]
    bbox = [273.9, 489.88, 86.39, 57.18]
    segmentation = [273.67, 489.38, 280.31, 538.02, 360.03, 547.37, 353.94, 498.73]
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]

    lp_image = resize_rectify(image, bbox, segmentation)
    cv2.imshow("LP", lp_image)
    cv2.imwrite("..//img//1.jpg", lp_image)

    lp_str = get_lp(lp_number)
    bbsoi = BoundingBoxesOnImage([
        BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
    ], shape=image.shape)
    kps = [Keypoint(x=segmentation[0], y=segmentation[1]),
           Keypoint(x=segmentation[2], y=segmentation[3]),
           Keypoint(x=segmentation[4], y=segmentation[5]),
           Keypoint(x=segmentation[6], y=segmentation[7])
           ]
    kpsoi = KeypointsOnImage(kps, shape=image.shape)
    image1 = kpsoi.draw_on_image(image, size=8, color=(0, 0, 255))
    image2 = bbsoi.draw_on_image(image1, size=3)

    cv2.rectangle(image2, (int(bbox[0]), int(bbox[1]) - 43), (int(bbox[0]) + 160, int(bbox[1]) - 2), (255, 255, 255),
                  thickness=-1)
    img_PIL = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('simhei.ttf', 40)
    fillColor = (255, 0, 0)
    position = (int(bbox[0] + 2), int(bbox[1]) - 40)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, lp_str, font=font, fill=fillColor)
    image2 = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)

    cv2.imshow("pre", image2)
    cv2.imwrite("..//img//pre1.jpg", image2)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
