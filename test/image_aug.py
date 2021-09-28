import imageio
import numpy as np
import imgaug as ia
import cv2 as cv
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

# ia.seed(1)

# image = imageio.imread("F:\\code_download\\CCPD2019\\ccpd_base\\0205399904214-90_93-254&457_494&528-494&532_256&537_260&457_498&452-0_0_21_32_3_32_29-125-23.jpg")
#
# print("Original:")
# ia.imshow(image)

image = cv.imread(
    "F:\\code_download\\CCPD2019\\ccpd_base\\0205399904214-90_93-254&457_494&528-494&532_256&537_260&457_498&452-0_0_21_32_3_32_29-125-23.jpg")[
        :, :, ::-1]
# image = cv.resize(image, (360, 580))
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=254, y1=457, x2=497, y2=528)
], shape=image.shape)
kps = [Keypoint(x=494, y=532),
       Keypoint(x=256, y=537),
       Keypoint(x=260, y=457),
       Keypoint(x=498, y=452)
       ]
kpsoi = KeypointsOnImage(kps, shape=image.shape)
image1 = kpsoi.draw_on_image(image, size=7, color=(255, 0, 0))
# ia.imshow(bbs.draw_on_image(image1, size=2))

seq = iaa.Sequential([
    iaa.GammaContrast(1.5),
    iaa.Affine(translate_percent={"x": (0.0, 0.2), "y": (0.0, 0.2)}, scale=0.8)
])
image_aug2, bbs_aug2, kpsoi_aug = seq(image=image, bounding_boxes=bbs, keypoints=kpsoi)
image2 = kpsoi_aug.draw_on_image(image_aug2, size=7, color=(255, 0, 0))
# ia.imshow(bbs_aug.draw_on_image(image1, size=2))

image_aug3, bbs_aug3, kpsoi_aug = iaa.Affine(rotate=15)(image=image, bounding_boxes=bbs, keypoints=kpsoi)
image3 = kpsoi_aug.draw_on_image(image_aug3, size=7, color=(255, 0, 0))
# ia.imshow(bbs_aug.draw_on_image(image1, size=2))

image_aug4, bbs_aug4, kpsoi_aug = iaa.PerspectiveTransform(scale=(0.01, 0.15))(image=image, bounding_boxes=bbs,
                                                                               keypoints=kpsoi)
image4 = kpsoi_aug.draw_on_image(image_aug4, size=7, color=(255, 0, 0))
# ia.imshow(bbs_aug.draw_on_image(image1, size=2))
ia.imshow(np.hstack([
    bbs.draw_on_image(image1, size=2),
    bbs_aug2.draw_on_image(image2, size=2),
    bbs_aug3.draw_on_image(image3, size=2),
    bbs_aug4.draw_on_image(image4, size=2)
]))

if any((abs(kpsoi_aug[0].x - kpsoi_aug[3].x) < 4, abs(kpsoi_aug[1].x - kpsoi_aug[2].x) < 4,
        abs(kpsoi_aug[0].y - kpsoi_aug[1].y) < 4, abs(kpsoi_aug[2].y - kpsoi_aug[3].y) < 4)):
    image_aug = image
    bbs_aug = bbs
    kpsoi_aug = kpsoi

print('---------------')
# print("Original:")
# ia.imshow(image)
# # cv.imshow("original", image)
#
# rotate = iaa.Affine(rotate=(-45, 45))
# image_aug = rotate(image=image)
# print("Augmented:")
# ia.imshow(image_aug)
