from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
dataDir = 'F:\code_download\CCPD2019'
dataType = 'train2020'
annFile = '{}/annotations/ccpd_base_{}.json'.format(dataDir, dataType)
# 初始化标注数据的 COCO api
coco = COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

h, w = image.shape[0], image.shape[1]
left = min(bbox[0], segmentation[0], segmentation[2])
rigth = max(bbox[2], segmentation[4], segmentation[6])
top = min(bbox[1], segmentation[1], segmentation[7])
bottom = max(bbox[3], segmentation[3], segmentation[5])
# 适当扩大10％的裁切范围，如果车牌区域太大则扩大固定范围

padding_w = int((rigth - left) * 0.1) if ((rigth - left) * 0.1) < 44.8 else 44
padding_h = int((bottom - top) * 0.1) if ((bottom - top) * 0.1) < 12.8 else 12

