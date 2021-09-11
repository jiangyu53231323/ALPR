import os

import imageio
import math
import numpy as np
import imgaug as ia
import cv2 as cv
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import pycocotools.coco as coco
from utils.my_image import resize_and_padding

coco = coco.COCO("../ccpd_val2020.json")
img_dir = "F:\\code_download\\CCPD2019\\ccpd_base"
images = coco.getImgIds()
img_id = images[6]
img_path = os.path.join(img_dir, coco.loadImgs(ids=[img_id])[0]['file_name'])
ann_ids = coco.getAnnIds(imgIds=img_id)
annotations = coco.loadAnns(ids=ann_ids)
labels = np.array([anno['category_id'] for anno in annotations])
bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32).squeeze()  # 降维
segmentation = np.array([anno['segmentation'] for anno in annotations], dtype=np.float32).squeeze()
print(img_path)
print(labels)
print(bboxes)
print(segmentation)

image = cv.imread(img_path)[:, :, ::-1]

image, scale, bboxes, segmentation = resize_and_padding(image, 256, bboxes, segmentation)

ia.imshow(image)

# # ------图像resize------
# height, width = image.shape[0], image.shape[1]  # 获取原始图片尺寸
# center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
# scale = max(height, width) * 1.0  # 长边计算缩放比例
# scale = scale / 512
# new_height = round(height / scale)
# new_width = round(width / scale)
# image = cv.resize(image, (new_width, new_height))
# # 判断resize后的图片能否被4整除，如果不能则要进行填充
# padding_h, padding_w = 0, 0
# if new_height % 4 != 0:
#     padding_h = new_height % 4
# if new_width % 4 != 0:
#     padding_w = new_width % 4
# new_image = np.zeros((new_height + padding_h, new_width + padding_w, 3), dtype=np.uint8)
# new_image[padding_h // 2:padding_h // 2 + new_height, padding_w // 2:padding_w // 2 + new_width] = image
# new_image[0:padding_h // 2, :, :] = 128
# new_image[:, 0:padding_w // 2, :] = 128
# new_image[padding_h // 2 + new_height:, :, :] = 128
# new_image[:, padding_w // 2 + new_width:, :] = 128
#
# image = new_image

# new_size = (image.shape[1] // 8, image.shape[0] // 8)
# image = cv.resize(image, new_size)
# ia.imshow(image)

if len(bboxes) == 0:
    bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
    labels = np.array([[0]])
# bboxes[2:] += bboxes[:2]  # xywh to xyxy
# 标签同步缩放
# bboxes = bboxes / scale
# segmentation = segmentation / scale

bbs = BoundingBoxesOnImage([
    BoundingBox(x1=bboxes[0], y1=bboxes[1], x2=bboxes[2], y2=bboxes[3])
], shape=image.shape)
# bbs = BoundingBox(x1=bboxes[0], y1=bboxes[1], x2=bboxes[0] + bboxes[2], y2=bboxes[1] + bboxes[3])
# 角点坐标 从左上角开始逆时针
kps = [Keypoint(x=segmentation[0], y=segmentation[1]),
       Keypoint(x=segmentation[2], y=segmentation[3]),
       Keypoint(x=segmentation[4], y=segmentation[5]),
       Keypoint(x=segmentation[6], y=segmentation[7])
       ]
kpsoi = KeypointsOnImage(kps, shape=image.shape)
# image1 = kpsoi.draw_on_image(image, size=7, color=(255, 0, 0))
# ia.imshow(bbs.draw_on_image(image1, size=2))

seq = iaa.Sequential([
    # iaa.GammaContrast(1.5),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-15, 15), scale=1),
    iaa.PerspectiveTransform(scale=(0.01, 0.15))
])
image_aug, bbs_aug, kpsoi_aug = seq(image=image, bounding_boxes=bbs, keypoints=kpsoi)
print(image_aug.shape)
print(bbs_aug)
print(kpsoi_aug)

print(kpsoi_aug[3].is_out_of_image(image_aug))
print(bbs_aug[0].is_fully_within_image(image_aug))

# 包围盒、角点越界判断，若产生越界则不做图像增强处理
if any((not bbs_aug[0].is_fully_within_image(image_aug), kpsoi_aug[0].is_out_of_image(image_aug),
        kpsoi_aug[1].is_out_of_image(image_aug), kpsoi_aug[2].is_out_of_image(image_aug),
        kpsoi_aug[3].is_out_of_image(image_aug))):
    image_aug = image
    bbs_aug = bbs
    kpsoi_aug = kpsoi

# x_min = min(kpsoi_aug[1].x, kpsoi_aug[2].x, kpsoi_aug[0].x, kpsoi_aug[3].x)
# x_max = max(kpsoi_aug[0].x, kpsoi_aug[3].x, kpsoi_aug[1].x, kpsoi_aug[2].x)
# y_min = min(kpsoi_aug[2].y, kpsoi_aug[3].y, kpsoi_aug[0].y, kpsoi_aug[1].y)
# y_max = max(kpsoi_aug[0].y, kpsoi_aug[1].y, kpsoi_aug[2].y, kpsoi_aug[3].y)
# if any((x_min < 0, x_max >= kpsoi_aug.shape[1], y_min < 0, y_max >= kpsoi_aug.shape[0], bbs_aug[0].x1 < 0,
#         bbs_aug[0].x2 >= bbs.shape[1], bbs_aug[0].y1 < 0, bbs_aug[0].y2 >= bbs.shape[0])):
#     image_aug = image
#     bbs_aug = bbs
#     kpsoi_aug = kpsoi

# 寻找四边形中心  将四个角点的横纵坐标相加再除以4
# ceil向上取整
center_x = math.ceil((kpsoi_aug[0].x + kpsoi_aug[1].x + kpsoi_aug[2].x + kpsoi_aug[3].x) / 4.0)
center_y = math.ceil((kpsoi_aug[0].y + kpsoi_aug[1].y + kpsoi_aug[2].y + kpsoi_aug[3].y) / 4.0)
center = [center_x, center_y]
print(center)
kps_c = [Keypoint(x=center_x, y=center_y)]
kpsoi_c = KeypointsOnImage(kps_c, shape=image.shape)
image1 = kpsoi_c.draw_on_image(image_aug, size=7, color=(255, 0, 0))
image1 = kpsoi_aug.draw_on_image(image1, size=7, color=(255, 0, 0))
ia.imshow(bbs_aug.draw_on_image(image1, size=2))

# 寻找四边形的几何属性（倾斜角度、长边、短边）
x1 = kpsoi_aug[0].x
y1 = kpsoi_aug[0].y
x2 = kpsoi_aug[1].x
y2 = kpsoi_aug[1].y
x3 = kpsoi_aug[2].x
y3 = kpsoi_aug[2].y
x4 = kpsoi_aug[3].x
y4 = kpsoi_aug[3].y
# 长、短边长
l1 = math.sqrt(pow((x4 - x1), 2) + pow((y4 - y1), 2))
l2 = math.sqrt(pow((x3 - x2), 2) + pow((y3 - y2), 2))
l_side = (l1 + l2) / 2
print('l_side:' + str(l_side))
s1 = math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
s2 = math.sqrt(pow((x3 - x4), 2) + pow((y3 - y4), 2))
s_side = (s1 + s2) / 2
print('s_side:' + str(s_side))
# 角度,单位为弧度
angle1 = math.atan2((y4 - y1), (x4 - x1))
angle2 = math.atan2((y3 - y2), (x3 - x2))
angle = (angle2 + angle1) / 2
print(angle / math.pi * 180)

'''
高斯分布概率
'''
# 高斯分布概率
# 对边界进行约束，防止越界
left, right = math.ceil(min(x1, x2)), math.ceil(max(x3, x4))  # 向上取整
top, bottom = math.ceil(min(y1, y4)), math.ceil(max(y2, y3))
# m = int((l_side - 1.) / 2)
# n = int((s_side - 1.) / 2)
# x, y = np.ogrid[-m:m + 1, -n:n + 1]
x, y = np.ogrid[left - center_x:right - center_x + 1, top - center_y:bottom - center_y + 1]
sigma_x = 0.54 * l_side / 24
sigma_y = 0.54 * s_side / 24
# 旋转矩阵
rotation_matrix = np.array([[math.cos(angle), -(math.sin(angle))], [math.sin(angle), math.cos(angle)]])
# rotation_matrix_I = np.linalg.inv(rotation_matrix)
# 缩放矩阵,对角线元素为长短轴长度×比例
scaling_matrix = np.array([[0.54 * l_side / 4, 0], [0, 0.54 * s_side / 4]])
# 协方差矩阵
sigma = np.dot(rotation_matrix, scaling_matrix)
sigma = np.linalg.inv(np.dot(sigma, sigma.T))
# sigma = np.linalg.multi_dot([rotation_matrix, scaling_matrix, scaling_matrix.T, np.linalg.inv(rotation_matrix)])

vector = np.array([x, y], dtype=object)
# res = -0.5 * np.dot(np.dot(vector, sigma), vector.T)
h = np.exp(-0.5 * np.dot(np.dot(vector, sigma), vector.T))
# h = np.exp(-(x * x / (2 * sigma_x * sigma_x)) - (y * y / (2 * sigma_y * sigma_y)))
# 限制最小值
# h[h < np.finfo(h.dtype).eps * h.max()] = 0
h[h < 0.01] = 0

heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
masked_heatmap = heatmap[top:bottom + 1, left:right + 1]
masked_gaussian = h.T
plt.imshow(masked_gaussian)
plt.show()
if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    # 将高斯分布覆盖到 heatmap 上，相当于不断的在 heatmap 基础上添加关键点的高斯，
    # 即同一种类型的框会在一个 heatmap 某一个类别通道上面上面不断添加。
    # 最终通过函数总体的 for 循环，相当于不断将目标画到 heatmap
    np.maximum(masked_heatmap, masked_gaussian * 1, out=masked_heatmap)
# 显示热力图
plt.figure(dpi=100, figsize=(12, 8))
plt.imshow(heatmap)
plt.show()

'''
角点坐标
'''
# 在高斯分布上标注角点坐标
corner = np.zeros((image.shape[0], image.shape[1], 8), dtype=np.float32)
masked_corner = corner[top:bottom + 1, left:right + 1]
# 在蒙版上定位中心点
corner_center_x = center_x - left
corner_center_y = center_y - top

for i in range(masked_corner.shape[0]):
    for j in range(masked_corner.shape[1]):
        masked_corner[i][j] = [j + left - x1, i + top - y1, j + left - x2, -(i + top - y2), -(j + left - x3),
                               -(i + top - y3), -(j + left - x4), i + top - y4]
masked_corner = masked_corner / 16
masks = masked_gaussian
masks[masks != 0] = 1
masks = np.expand_dims(masks, axis=-1)
masked_corner = masked_corner * masks

inds = np.zeros((1,), dtype=np.int64)
inds[0] = center_y * image.shape[1] + center_x
print(inds)

print(masked_corner)
