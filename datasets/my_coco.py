import copy
import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug import augmenters as iaa

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.my_image import resize_and_padding, image_affine, draw_heatmap_gaussian, draw_corner_gaussian, \
    draw_bboxes_gaussian, new_image_affine
from utils.utils import get_image_path

COCO_NAMES = ['__background__', 'License Plate']
COCO_IDS = [1]
# CCPD all 数据集的平均值和标准差,RGB顺序
COCO_MEAN = [0.41250753, 0.46157351, 0.4346494]
COCO_STD = [0.2468385, 0.2403439, 0.24371054]
# 作用未知，在仿射变换中使用
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]
# 控制高斯分布半径的缩放系数，数值越小半径越小
GAUSSIAN_SCALE = 0.12


class COCO(data.Dataset):
    def __init__(self, data_dir, split, split_ratio=1.0, img_size=384):
        super(COCO, self).__init__()
        # 数据集类别总数、类别名、有效ID
        self.num_classes = len(COCO_IDS)
        self.class_name = COCO_NAMES
        self.valid_ids = COCO_IDS
        # cat_ids 将类别list转换成字典形式
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}
        # 伪随机数生成器
        self.data_rng = np.random.RandomState(123)

        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.gaussian_scale = GAUSSIAN_SCALE
        # 数据集 均值和方差
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]
        # split用来区分train或val
        self.split = split
        # 数据集路径 data/CCPD2019
        # 图片 data/CCPD2019/ccpd_base
        self.data_dir = os.path.join(data_dir, 'CCPD2019')
        self.img_dir = os.path.join(self.data_dir, 'ccpd')
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'ccpd_%s2020.json' % split)
        # self.annot_path = os.path.join('./', 'ccpd_%s2020.json' % split)

        self.max_objs = 1  # 最大检测目标数
        self.padding = 127  # 31 for resnet/resdcn
        self.down_ratio = 4  # 特征图缩小倍数
        # 缩放后的image大小
        self.img_size = {'h': img_size, 'w': img_size}
        # 特征图大小
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}  # // 为向下取整
        self.rand_scales = np.arange(0.6, 1.4, 0.1)  # [0.6,0.7,0.8,...,1.2,1.3]
        self.gaussian_iou = 0.7  #

        print('==> initializing CCPD 2019 %s data.' % split)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()

        # 在所有数据集中取出split_ratio比例的示例
        if 0 < split_ratio < 1:
            split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
            self.images = self.images[:split_size]
        # 获取总样本数
        self.num_samples = len(self.images)

        print('Loaded %d %s samples' % (self.num_samples, split))

    def __getitem__(self, index):
        # 根据index得到image对应的id，再由id得到图片文件名，拼接成路径
        img_id = self.images[index]
        # img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])

        # 如果 self.img_dir 下有多个文件夹，则使用get_image_path来寻找image的真正路径
        img_path = get_image_path(self.data_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])

        # 根据image id 获取 annotion id
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)

        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32).squeeze()  # 降维
        segmentation = np.array([anno['segmentation'] for anno in annotations], dtype=np.float32).squeeze()  # 降维
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])
        bboxes[2:] += bboxes[:2]  # xywh to xyxy
        # 读取图片
        image = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
        # 调整图片大小并填充，返回调整后的图片和缩小的比例
        # image, scale, bboxes, segmentation = resize_and_padding(image, self.img_size['h'], bboxes, segmentation)
        resize_out = resize_and_padding(image, self.img_size['h'], bboxes, segmentation)
        image = resize_out['new_image']
        scale = resize_out['scale']
        bboxes = resize_out['bboxes']
        segmentation = resize_out['segmentation']
        padding_h = resize_out['padding_h']
        padding_w = resize_out['padding_w']

        flipped = False  # 翻转

        # 随机 仿射+透视 变换， img_aug参数是是否做数据增强，False为不增强
        # image, bbs, kpsoi = image_affine(image, bboxes, segmentation, img_id, img_aug=True)
        # image, bboxes, segmentation = new_image_affine(image, bboxes, segmentation, img_id)
        # ---------------------------------------------------------------------------------
        image = image.astype(np.float32) / 255.
        image -= self.mean
        image /= self.std
        image = image.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        fmap_h = image.shape[1]
        fmap_w = image.shape[2]
        # heatmap
        heat_map = np.zeros((self.num_classes, math.ceil(fmap_h / self.down_ratio),
                             math.ceil(fmap_w / self.down_ratio)), dtype=np.float32)
        '''
        corner是四个角点的标注，按照【max_objs,h,w,8】格式，则网络输出与之无法对应，因为网络输出的格式为【c,8,h,w】，
        这时max_objs就是一个多余的维度，所以需要去掉。
        对于同类多目标的角点标注，可以将每个物体的标注依次覆盖到【c,8,h,w】上，先将大目标画到corner，再将小目标画上去，
        这样做的好处是如果物体间有重合的地方，则能够保留较小目标的全部特征。
        '''
        # 角点坐标
        # corner = np.zeros((self.max_objs, math.ceil(fmap_h / self.down_ratio),
        #                    math.ceil(fmap_w / self.down_ratio), 8), dtype=np.float32)
        corner_map = np.zeros((8, math.ceil(fmap_h / self.down_ratio), math.ceil(fmap_w / self.down_ratio)),
                              dtype=np.float32)
        bboxes_map = np.zeros((4, math.ceil(fmap_h / self.down_ratio), math.ceil(fmap_w / self.down_ratio)),
                              dtype=np.float32)

        # 目标中心点在特征图上的序号，行优先排列
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        # inds_masks标记inds中的元素是否有目标，如inds中有[x,0,0,0] inds_masks[1,0,0,0] 则说明只有x位置是存在目标的，0位置是无目标的初始元素
        # 在这里是单目标检测，情况会简单很多
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
        # 将高斯分布画到heatmap上
        masked_gaussian, center = draw_heatmap_gaussian(heat_map[0], segmentation, self.gaussian_scale,
                                                        self.down_ratio)
        draw_corner_gaussian(corner_map, segmentation, masked_gaussian, self.down_ratio)
        draw_bboxes_gaussian(bboxes_map, bboxes, segmentation, masked_gaussian, self.down_ratio)
        # inds保存heatmap中目标点的索引，也就是正样本的位置索引
        inds[0] = center[1] * heat_map.shape[1] + center[0]
        ind_masks[0] = 1

        # 设置角点和边界框loss计算的的mask
        reg_mask = copy.deepcopy(heat_map)
        # reg_mask[reg_mask != 0] = 1
        # corner_mask = np.zeros((8, math.ceil(fmap_h / self.down_ratio), math.ceil(fmap_w / self.down_ratio)),
        #                        dtype=np.float32)
        # bboxes_mask = np.zeros((4, math.ceil(fmap_h / self.down_ratio), math.ceil(fmap_w / self.down_ratio)),
        #                        dtype=np.float32)
        # for ind in range(8):
        #     corner_mask[ind] = reg_mask[0]
        # for ind in range(4):
        #     bboxes_mask[ind] = reg_mask[0]

        return {'image': image, 'hmap': heat_map, 'corner': corner_map, 'bboxes': bboxes_map, 'inds': inds,
                'ind_masks': ind_masks, 'scale': scale, 'img_id': img_id, 'reg_mask': reg_mask}

    def __len__(self):
        return self.num_samples


class COCO_eval(COCO):
    def __init__(self, data_dir, split, test_scales=(1,), test_flip=False, fix_size=True, **kwargs):
        super(COCO_eval, self).__init__(data_dir, split)
        self.test_flip = test_flip  #
        self.test_scales = test_scales
        self.fix_size = fix_size

    def __getitem__(self, index):
        # 根据index得到image对应的id，再由id得到图片文件名，拼接成路径
        img_id = self.images[index]
        # img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        # 如果 self.img_dir 下有多个文件夹，则使用get_image_path来寻找image的真正路径
        img_path = get_image_path(self.data_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        # 根据image id 获取 annotion id
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)

        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32).squeeze()  # 降维
        segmentation = np.array([anno['segmentation'] for anno in annotations], dtype=np.float32).squeeze()  # 降维
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])
        bboxes[2:] += bboxes[:2]  # xywh to xyxy

        image = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
        height, width = image.shape[0:2]

        out = {}
        for scale in self.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)
            new_bboxes = bboxes * scale
            new_segmentation = segmentation * scale

            if self.fix_size:
                img_height, img_width = self.img_size['h'], self.img_size['w']
                # center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                # scaled_size = max(height, width) * 1.0
                # scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
            else:
                img_height = (new_height | self.padding) + 1
                img_width = (new_width | self.padding) + 1
                # center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                # scaled_size = np.array([img_width, img_height], dtype=np.float32)

            resize_out = resize_and_padding(image, max(img_height, img_width), new_bboxes, new_segmentation)
            img = resize_out['new_image']
            img_scale = resize_out['scale']
            new_bboxes = resize_out['bboxes']
            new_segmentation = resize_out['segmentation']
            padding_h = resize_out['padding_h']
            padding_w = resize_out['padding_w']
            # img, img_scale, new_bboxes, new_segmentation = resize_and_padding(image, max(img_height, img_width),
            #                                                                   new_bboxes, new_segmentation)
            img_height = img.shape[0]
            img_width = img.shape[1]
            # trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
            # img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            img = img.astype(np.float32) / 255.
            img -= self.mean
            img /= self.std
            img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

            if self.test_flip:  # 翻转图片
                img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

            out[scale] = {'image': img,
                          'fmap_h': img_height // self.down_ratio,
                          'fmap_w': img_width // self.down_ratio,
                          'bboxes': new_bboxes,
                          'segmentation': new_segmentation,
                          'image_scale': img_scale,
                          'padding_w': padding_w,
                          'padding_h': padding_h
                          }

        return img_id, out

    def convert_eval_format(self, all_bboxes):
        # all_bboxes: num_samples x num_classes x 5
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self.valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

                    detection = {"image_id": int(image_id),
                                 "category_id": int(category_id),
                                 "bbox": bbox_out,
                                 "score": float("{:.2f}".format(score))}
                    detections.append(detection)
        return detections

    def run_eval(self, results, save_dir=None):
        # 转变格式，使之能使用coco api 计算map
        detections = self.convert_eval_format(results)

        if save_dir is not None:
            result_json = os.path.join(save_dir, "results.json")
            json.dump(detections, open(result_json, "w"))
        # 使用COCO api进行AP计算
        coco_dets = self.coco.loadRes(detections)
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            # 将image从array转换为tensor
            out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
            if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
        return out
