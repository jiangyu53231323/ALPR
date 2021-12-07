import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

a = cv.imread(
    'E:\\CodeDownload\\data\\CCPD2019\\ccpd/01-89_84-231&482_413&539-422&537_242&536_219&475_399&476-0_0_15_11_25_26_26-121-16.jpg')
# a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
bbox = [231, 482, 413, 539]
segmentation = [219, 475, 242, 536, 422, 537, 399, 476]
left = min(bbox[0], segmentation[0], segmentation[2])
rigth = max(bbox[2], segmentation[4], segmentation[6])
top = min(bbox[1], segmentation[1], segmentation[7])
bottom = max(bbox[3], segmentation[3], segmentation[5])

a = a[top:bottom + 1, left:rigth + 1, :]


pts = np.float32([[segmentation[0] - left, segmentation[1] - top], [segmentation[2] - left, segmentation[3] - top],
                  [segmentation[4] - left, segmentation[5] - top], [segmentation[6] - left, segmentation[7] - top]])

pts1 = np.float32(
    [[10, 10], [10, bottom - top - 10], [rigth - left - 10, bottom - top - 10], [rigth - left - 10, 10]])

M = cv.getPerspectiveTransform(pts, pts1)

dst = cv.warpPerspective(a, M, (rigth - left + 1,bottom - top + 1))

cv.imshow('original', a)
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
