import math
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


def batch_resize_rectify(image, bbox, segmentation, size=(224, 64), is_rectify=True):
    '''
    :param image:[h,w,c]
    :param bbox: [x1,y1,x2,y2]
    :param segmentation: [x1,y1,x2,y2,x3,y3,x4,y4]
    :param size: (w,h)
    :return: [b,c,h,w]
    '''
    # image = batch[0].transpose(1, 2, 0)  # permute可以操作多维，transpose只能是2维
    h, w = image.shape[0], image.shape[1]
    left = min(bbox[0], segmentation[0], segmentation[2])
    rigth = max(bbox[2], segmentation[4], segmentation[6])
    top = min(bbox[1], segmentation[1], segmentation[7])
    bottom = max(bbox[3], segmentation[3], segmentation[5])
    # 适当扩大10％的裁切范围，四个方向各10%
    padding_w = int((rigth - left) * 0.1)
    padding_h = int((bottom - top) * 0.1)
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
            [[16, 5], [16, 60], [208, 60], [208, 5]])

        M = cv2.getPerspectiveTransform(pts, pts1)
        dst = cv2.warpPerspective(image_new, M, size)

    dst = dst.transpose(2, 0, 1)[None, :, :, :]
    dst_tensor = torch.from_numpy(dst)

    return dst_tensor


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
    # 适当扩大10％的裁切范围
    padding_w = int((rigth - left) * 0.1)
    padding_h = int((bottom - top) * 0.1)
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
            [[16, 5], [16, 60], [208, 60], [208, 5]])

        M = cv2.getPerspectiveTransform(pts, pts1)
        dst = cv2.warpPerspective(image_new, M, size)
        return dst
    else:
        return image_new


def resize_and_padding(image, size, bboxes, segmentation, pre_384=False):
    '''

    :param image:
    :param size:
    :param bboxes: [x1,y1,x2,y2]
    :param segmentation:
    :param pre_384:
    :return:
    '''
    if pre_384 == True:
        scale = 1160 / size
        new_height = math.ceil(1160 / scale)
        new_width = math.ceil(720 / scale)
    else:
        # ------图像resize------
        height, width = image.shape[0], image.shape[1]  # 获取原始图片尺寸
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0  # 长边计算缩放比例
        scale = scale / size
        new_height = math.ceil(height / scale)
        new_width = math.ceil(width / scale)
        image = cv2.resize(image, (new_width, new_height))
    # 判断resize后的图片能否被32整除，如果不能则要进行填充
    padding_h, padding_w = 0, 0
    if new_height % 32 != 0:
        padding_h = 32 - (new_height % 32)
    if new_width % 32 != 0:
        padding_w = 32 - (new_width % 32)
    # 初始化new_image时使用固定值=128，省去下方填充像素的4行代码
    new_image = np.zeros((new_height + padding_h, new_width + padding_w, 3), dtype=np.uint8)
    new_image[padding_h // 2:padding_h // 2 + new_height, padding_w // 2:padding_w // 2 + new_width] = image
    new_image[0:padding_h // 2, :, :] = 128
    new_image[:, 0:padding_w // 2, :] = 128
    new_image[padding_h // 2 + new_height:, :, :] = 128
    new_image[:, padding_w // 2 + new_width:, :] = 128

    # 标签同步缩放
    bboxes = bboxes / scale
    segmentation = segmentation / scale
    bboxes = np.array([bboxes[0] + (padding_w // 2), bboxes[1] + (padding_h // 2), bboxes[2] + (padding_w // 2),
                       bboxes[3] + (padding_h // 2), ])
    segmentation = np.array([segmentation[0] + (padding_w // 2), segmentation[1] + (padding_h // 2),
                             segmentation[2] + (padding_w // 2), segmentation[3] + (padding_h // 2),
                             segmentation[4] + (padding_w // 2), segmentation[5] + (padding_h // 2),
                             segmentation[6] + (padding_w // 2), segmentation[7] + (padding_h // 2), ])
    out = {'new_image': new_image, 'scale': scale, 'bboxes': bboxes, 'segmentation': segmentation,
           'padding_h': padding_h, 'padding_w': padding_w}
    return out


def new_image_affine(image, bboxes, segmentation, img_id):
    bbox = [[bboxes[0], bboxes[1], bboxes[2], bboxes[3]], ]
    keypoints = [(segmentation[0], segmentation[1]),
                 (segmentation[2], segmentation[3]),
                 (segmentation[4], segmentation[5]),
                 (segmentation[6], segmentation[7]), ]
    category_ids = [1]
    category_id_to_name = {1: 'licence plate'}

    transform = A.Compose([
        # A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5,
                           border_mode=cv2.BORDER_REPLICATE, ),
        A.Perspective(scale=(0.05, 0.15), p=0.25), ],
        bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.6, label_fields=['category_ids']),
        keypoint_params=A.KeypointParams(format='xy'))
    transformed = transform(image=image, bboxes=bbox, keypoints=keypoints, category_ids=category_ids)

    if len(transformed['keypoints']) < 4 or len(transformed['bboxes']) < 1:
        # print('数据增强导致目标越界，取消增强')
        return image, bboxes, segmentation
    else:
        keypoints_aug = []
        for (x, y) in transformed['keypoints']:
            keypoints_aug.append(x)
            keypoints_aug.append(y)
        return transformed['image'], np.array(list(transformed['bboxes'][0])), np.array(keypoints_aug)


# 仿射+透视变换
def image_affine(image, bboxes, segmentation, img_id, img_aug=True):
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=bboxes[0], y1=bboxes[1], x2=bboxes[2], y2=bboxes[3])
    ], shape=image.shape)
    # 角点坐标 从左上角开始逆时针顺序
    kps = [Keypoint(x=segmentation[0], y=segmentation[1]),
           Keypoint(x=segmentation[2], y=segmentation[3]),
           Keypoint(x=segmentation[4], y=segmentation[5]),
           Keypoint(x=segmentation[6], y=segmentation[7])
           ]
    kpsoi = KeypointsOnImage(kps, shape=image.shape)
    # 是否做数据增强
    if img_aug == True:
        # 平移+旋转
        image_aug, bbs_aug, kpsoi_aug = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                                   rotate=(-15, 15),
                                                   scale=1)(image=image, bounding_boxes=bbs, keypoints=kpsoi)
        # 包围盒、角点越界判断，若产生越界则不做图像增强处理
        if any((not bbs_aug[0].is_fully_within_image(image_aug), kpsoi_aug[0].is_out_of_image(image_aug),
                kpsoi_aug[1].is_out_of_image(image_aug), kpsoi_aug[2].is_out_of_image(image_aug),
                kpsoi_aug[3].is_out_of_image(image_aug))):
            image_aug = image
            bbs_aug = bbs
            kpsoi_aug = kpsoi
        if any((abs(kpsoi_aug[0].x - kpsoi_aug[3].x) < 4, abs(kpsoi_aug[1].x - kpsoi_aug[3].x) < 4,
                abs(kpsoi_aug[0].y - kpsoi_aug[1].y) < 4, abs(kpsoi_aug[2].y - kpsoi_aug[3].y) < 4)):
            image_aug = image
            bbs_aug = bbs
            kpsoi_aug = kpsoi
        # 透视变换
        image_aug2, bbs_aug2, kpsoi_aug2 = iaa.PerspectiveTransform(scale=(0.01, 0.15))(image=image_aug,
                                                                                        bounding_boxes=bbs_aug,
                                                                                        keypoints=kpsoi_aug)

        # 包围盒、角点越界判断，若产生越界则不做图像增强处理
        if any((not bbs_aug2[0].is_fully_within_image(image_aug), kpsoi_aug2[0].is_out_of_image(image_aug),
                kpsoi_aug2[1].is_out_of_image(image_aug), kpsoi_aug2[2].is_out_of_image(image_aug),
                kpsoi_aug2[3].is_out_of_image(image_aug))):
            image_aug2 = image_aug
            bbs_aug2 = bbs_aug
            kpsoi_aug2 = kpsoi_aug
        # 包围盒、角点异常判断，若对应点距离小于4则不做图像增强处理
        if any((abs(kpsoi_aug2[0].x - kpsoi_aug2[3].x) < 4, abs(kpsoi_aug2[1].x - kpsoi_aug2[3].x) < 4,
                abs(kpsoi_aug2[0].y - kpsoi_aug2[1].y) < 4, abs(kpsoi_aug2[2].y - kpsoi_aug2[3].y) < 4)):
            ia.imshow(image_aug2)
            image_aug2 = image_aug
            bbs_aug2 = bbs_aug
            kpsoi_aug2 = kpsoi_aug
    else:
        image_aug2 = image
        bbs_aug2 = bbs
        kpsoi_aug2 = kpsoi

    return image_aug2, bbs_aug2, kpsoi_aug2


def draw_heatmap_gaussian(heatmap, seg, scale, down_ratio):
    # 计算缩小down_ratio后的车牌中心点坐标
    center_x = math.ceil((seg[0] + seg[2] + seg[4] + seg[6]) / (4.0 * down_ratio))
    center_y = math.ceil((seg[1] + seg[3] + seg[5] + seg[7]) / (4.0 * down_ratio))
    center = [center_x, center_y]
    # 寻找四边形的几何属性（倾斜角度、长边、短边）
    x1 = seg[0] / down_ratio
    y1 = seg[1] / down_ratio
    x2 = seg[2] / down_ratio
    y2 = seg[3] / down_ratio
    x3 = seg[4] / down_ratio
    y3 = seg[5] / down_ratio
    x4 = seg[6] / down_ratio
    y4 = seg[7] / down_ratio
    # 长、短边长
    l1 = math.sqrt(pow((x4 - x1), 2) + pow((y4 - y1), 2))
    l2 = math.sqrt(pow((x3 - x2), 2) + pow((y3 - y2), 2))
    l_side = (l1 + l2) / 2
    s1 = math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))
    s2 = math.sqrt(pow((x3 - x4), 2) + pow((y3 - y4), 2))
    s_side = (s1 + s2) / 2
    # 角度,单位为弧度
    angle1 = math.atan2((y4 - y1), (x4 - x1))
    angle2 = math.atan2((y3 - y2), (x3 - x2))
    angle = (angle2 + angle1) / 2
    # 高斯分布概率
    # 对边界进行约束，防止越界
    left, right = math.floor(min(x1, x2)), math.ceil(max(x3, x4))
    top, bottom = math.floor(min(y1, y4)), math.ceil(max(y2, y3))
    x, y = np.ogrid[left - center_x:right - center_x + 1, top - center_y:bottom - center_y + 1]
    # 旋转矩阵
    rotation_matrix = np.array([[math.cos(angle), -(math.sin(angle))], [math.sin(angle), math.cos(angle)]])
    # 缩放矩阵,对角线元素为长短轴长度×比例
    scaling_matrix = np.array([[scale * l_side, 0], [0, scale * s_side]])
    # 协方差矩阵
    sigma = np.dot(rotation_matrix, scaling_matrix)
    sigma = np.linalg.inv(np.dot(sigma, sigma.T))
    try:
        vector = np.array([x, y], dtype=object)
        h = np.exp(-0.5 * np.dot(np.dot(vector, sigma), vector.T))
        # 限制最小值
        h[h < 0.01] = 0
        masked_gaussian = h.T
    except:
        print('corner coordinate: x1=%f y1=%f x2=%f y2=%f x3=%f y3=%f x4=%f y4=%f' % (x1, y1, x2, y2, x3, y3, x4, y4))
        print('left: %f, right: %f, top: %f, bottom: %f' % (left, right, top, bottom))
        print('center_x: %f, center_y: %f' % (center_x, center_y))
        print(x)
        print(y)
        masked_gaussian = heatmap[top:bottom + 1, left:right + 1]

    masked_heatmap = heatmap[top:bottom + 1, left:right + 1]
    # mask = masks[top:bottom + 1, left:right + 1]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        # 将高斯分布覆盖到 heatmap 上，相当于不断的在 heatmap 基础上添加关键点的高斯，
        # 即同一种类型的框会在一个 heatmap 某一个类别通道上面上面不断添加。
        # 最终通过函数总体的 for 循环，相当于不断将目标画到 heatmap
        masked_gaussian = masked_gaussian[0:masked_heatmap.shape[0], 0:masked_heatmap.shape[1]]  # 数据对齐
        np.maximum(masked_heatmap, masked_gaussian * 1, out=masked_heatmap)
        # np.maximum(mask, masked_gaussian * 1, out=mask)
    masked_gaussian[masked_gaussian != 0] = 1
    masked_gaussian = np.expand_dims(masked_gaussian, axis=0)
    return masked_gaussian, center


def draw_corner_gaussian(corner, seg, masked_gaussian, down_ratio):
    x1 = seg[0] / down_ratio
    y1 = seg[1] / down_ratio
    x2 = seg[2] / down_ratio
    y2 = seg[3] / down_ratio
    x3 = seg[4] / down_ratio
    y3 = seg[5] / down_ratio
    x4 = seg[6] / down_ratio
    y4 = seg[7] / down_ratio
    # 对边界进行约束，防止越界
    left, right = math.floor(min(x1, x2)), math.ceil(max(x3, x4))
    top, bottom = math.floor(min(y1, y4)), math.ceil(max(y2, y3))
    # 在高斯分布上标注角点坐标
    corner_mask = corner[:, top:bottom + 1, left:right + 1]
    masked_corner = copy.deepcopy(corner_mask)  # masked_corner深拷贝corner_mask
    corner_mask[:, :, :] = -1e4
    center_x = math.ceil((x1 + x2 + x3 + x4) / 4.0)
    center_y = math.ceil((y1 + y2 + y3 + y4) / 4.0)
    # 在蒙版上定位中心点
    corner_center_x = center_x - left
    corner_center_y = center_y - top

    for i in range(masked_corner.shape[1]):
        for j in range(masked_corner.shape[2]):
            # masked_corner[i][j] = [j + left - x1, i + top - y1, j + left - x2, -(i + top - y2), -(j + left - x3),
            #                        -(i + top - y3), -(j + left - x4), i + top - y4]
            masked_corner[0][i][j] = j + left - x1
            masked_corner[1][i][j] = i + top - y1
            masked_corner[2][i][j] = j + left - x2
            masked_corner[3][i][j] = -(i + top - y2)
            masked_corner[4][i][j] = -(j + left - x3)
            masked_corner[5][i][j] = -(i + top - y3)
            masked_corner[6][i][j] = -(j + left - x4)
            masked_corner[7][i][j] = i + top - y4

    masked_corner = masked_corner * masked_gaussian  # / 8  # 8是一个放大系数，可以让网络预测的值保持在一个比较小的范围
    if min(masked_gaussian.shape) > 0 and min(masked_corner.shape) > 0:  # TODO debug
        # corner_mask = corner_mask * masked_corner
        np.maximum(corner_mask, masked_corner, out=corner_mask)
    return corner


def draw_bboxes_gaussian(bboxes_map, bbs, seg, masked_gaussian, down_ratio):
    x1 = seg[0] / down_ratio
    y1 = seg[1] / down_ratio
    x2 = seg[2] / down_ratio
    y2 = seg[3] / down_ratio
    x3 = seg[4] / down_ratio
    y3 = seg[5] / down_ratio
    x4 = seg[6] / down_ratio
    y4 = seg[7] / down_ratio
    # 对边界进行约束，防止越界
    left, right = math.floor(min(x1, x2)), math.ceil(max(x3, x4))
    top, bottom = math.floor(min(y1, y4)), math.ceil(max(y2, y3))

    # w1,h1 是左上角坐标，w2,h2是右下角坐标
    w1 = bbs[0] / down_ratio
    h1 = bbs[1] / down_ratio
    w2 = bbs[2] / down_ratio
    h2 = bbs[3] / down_ratio

    # 在高斯分布上标注角点坐标
    bboxes_mask = bboxes_map[:, top:bottom + 1, left:right + 1]
    masked_w_h_ = copy.deepcopy(bboxes_mask)  # masked_corner深拷贝corner_mask
    bboxes_mask[:, :, :] = -1e4

    center_x = math.ceil((x1 + x2 + x3 + x4) / 4.0)
    center_y = math.ceil((y1 + y2 + y3 + y4) / 4.0)
    # 在蒙版上定位中心点
    corner_center_x = center_x - left
    corner_center_y = center_y - top

    for i in range(masked_w_h_.shape[1]):
        for j in range(masked_w_h_.shape[2]):
            masked_w_h_[0][i][j] = j + left - w1
            masked_w_h_[1][i][j] = i + top - h1
            masked_w_h_[2][i][j] = -(j + left - w2)
            masked_w_h_[3][i][j] = -(i + top - h2)

    masked_corner = masked_w_h_ * masked_gaussian  # / 8  # 8是一个放大系数，可以让网络预测的值保持在一个比较小的范围
    if min(masked_gaussian.shape) > 0 and min(masked_corner.shape) > 0:  # TODO debug
        # corner_mask = corner_mask * masked_corner
        np.maximum(bboxes_mask, masked_corner, out=bboxes_mask)
    return bboxes_map


def flip(img):
    return img[:, :, ::-1].copy()


# todo what the hell is this?
def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


# 对预测结果进行仿射变换
def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


# 仿射变换：缩放+平移
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180  # 将rot角度换成弧度表示
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    # M = getAffineTransform(src,dst)
    # src:原始图像中的三个点坐标
    # dst:变换后的三个点对应的坐标
    # M:根据三个点求出的仿射变换矩阵
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


# 放射变换 坐标计算
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)  # 矩阵点乘
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


# 旋转变换角度
def get_dir(src_point, rot_rad):
    # np.sin(x) 对x元素取正弦
    _sin, _cos = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * _cos - src_point[1] * _sin
    src_result[1] = src_point[0] * _sin + src_point[1] * _cos

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


# 高斯半径 CornerNet
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    # 情况三
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    # r1 = (b1 + sq1) / 2 #
    r1 = (b1 - sq1) / (2 * a1)
    # 情况二
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    # r2 = (b2 + sq2) / 2
    r2 = (b2 - sq2) / (2 * a2)
    # 情况一
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    # r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


# 由直径得内切正方形高斯分布
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    # 高斯分布计算公式
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # 限制最小的值
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    # 得到直径
    diameter = 2 * radius + 1
    # sigma 是一个与直径相关的参数
    # 一个圆对应内切正方形的高斯分布
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    # 对边界进行约束，防止越界
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    # 选择对应区域 切片操作属于浅拷贝
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # 将高斯分布结果约束在边界内
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        # 将高斯分布覆盖到 heatmap 上，相当于不断的在 heatmap 基础上添加关键点的高斯，
        # 即同一种类型的框会在一个 heatmap 某一个类别通道上面上面不断添加。
        # 最终通过函数总体的 for 循环，相当于不断将目标画到 heatmap
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap
