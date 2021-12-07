import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

transform = A.Compose([
    A.Resize(64, 224)],
    keypoint_params=A.KeypointParams(format='xy'))

a = cv.imread(
    'E:\\CodeDownload\\data\\CCPD2019\\ccpd/01-89_84-231&482_413&539-422&537_242&536_219&475_399&476-0_0_15_11_25_26_26-121-16.jpg')
# a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
bbox = [231, 482, 413, 539]
segmentation = [219, 475, 242, 536, 422, 537, 399, 476]
left = min(bbox[0], segmentation[0], segmentation[2])
rigth = max(bbox[2], segmentation[4], segmentation[6])
top = min(bbox[1], segmentation[1], segmentation[7])
bottom = max(bbox[3], segmentation[3], segmentation[5])
# 适当扩大10％的裁切范围
padding_w = int((rigth - left) * 0.1)
padding_h = int((bottom - top) * 0.1)
left = left - padding_w
rigth = rigth + padding_w
top = top - padding_h
bottom = bottom + padding_h

a = a[top:bottom + 1, left:rigth + 1, :]
keypoints = [(segmentation[0] - left, segmentation[1] - top),
             (segmentation[2] - left, segmentation[3] - top),
             (segmentation[4] - left, segmentation[5] - top),
             (segmentation[6] - left, segmentation[7] - top), ]
category_ids = [1]
category_id_to_name = {1: 'PL'}
transformed = transform(image=a, keypoints=keypoints, category_ids=category_ids)
image_new = transformed['image']
keypoints_new = transformed['keypoints']

# pts = np.float32([[segmentation[0] - left, segmentation[1] - top], [segmentation[2] - left, segmentation[3] - top],
#                   [segmentation[4] - left, segmentation[5] - top], [segmentation[6] - left, segmentation[7] - top]])
pts = np.float32([[keypoints_new[0][0], keypoints_new[0][1]], [keypoints_new[1][0], keypoints_new[1][1]],
                  [keypoints_new[2][0], keypoints_new[2][1]], [keypoints_new[3][0], keypoints_new[3][1]]])

pts1 = np.float32(
    [[16, 5], [16, 60], [208, 60], [208, 5]])

M = cv.getPerspectiveTransform(pts, pts1)

dst = cv.warpPerspective(image_new, M, (224, 64))

cv.imshow('original', image_new)
cv.imshow('result', dst)
cv.waitKey(0)
cv.destroyAllWindows()

# plt.figure(figsize=(30, 30))
# plt.subplot(121)
# plt.imshow(a)
# plt.title('raw_img')
# plt.subplot(122)
# plt.imshow(dst)
# plt.title('tansform_img')
# plt.show()
