import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pycocotools.coco as coco

from utils.utils import get_image_path


class DataLoader(Dataset):
    def __init__(self, data_dir, img_size, split):
        super(DataLoader, self).__init__()
        self.split = split
        # 数据集路径 data/CCPD2019
        self.data_dir = os.path.join(data_dir, 'CCPD2019')
        self.img_dir = os.path.join(self.data_dir, 'ccpd')
        self.annot_path = os.path.join(self.data_dir, 'annotations', 'ccpd_%s2020.json' % split)
        self.img_size = img_size

        print('==> initializing CCPD 2019 %s data.' % split)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        # 获取总样本数
        self.num_samples = len(self.images)
        print('Loaded %d %s samples' % (self.num_samples, split))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = get_image_path(self.data_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        # 根据image id 获取 annotion id
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32).squeeze()  # 降维
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])
        bboxes[2:] += bboxes[:2]  # xywh to xyxy
        if bboxes[0] >= bboxes[2] or bboxes[1] >= bboxes[3]:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
        # 读取图片
        image = cv2.imread(img_path)[:, :, ::-1]  # BGR to RGB
        image = image[int(bboxes[1]):int(bboxes[3]) + 1, int(bboxes[0]):int(bboxes[2]) + 1, :]
        image = cv2.resize(image, self.img_size)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype('float32') / 255.
        img_name = self.coco.loadImgs(ids=[img_id])[0]['file_name'].split('.')[0]  # 分割图片名称，rsplit作用是去除.jpg后缀
        labels = [int(c) for c in img_name.split('-')[-3].split('_')[:7]]
        return image, labels

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            # 将image从array转换为tensor
            out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
            if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
        return out
