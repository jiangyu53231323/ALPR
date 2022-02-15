import PIL
from PIL import Image

import albumentations as A
import cv2
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White
KEYPOINT_COLOR = (0, 255, 0)  # Green


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=3):
    # image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    # plt.figure(figsize=(8, 8))
    # plt.axis('off')
    # plt.imshow(image)
    return image


def visualize(image, bboxes, keypoints, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    img = vis_keypoints(img, keypoints)
    plt.figure(figsize=(16, 16))
    plt.axis('off')
    plt.imshow(img)


def main():
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.6, scale_limit=0.2, rotate_limit=20, p=1, border_mode=cv2.BORDER_REPLICATE,
                           ),
        A.Perspective(scale=(0.05, 0.15), p=0.25), ],
        bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.9, label_fields=['category_ids']),
        keypoint_params=A.KeypointParams(format='xy'))

    image = cv2.imread(
        "E:\\CodeDownload\\data\\CCPD2019\ccpd\\0205399904214-90_93-254&457_494&528-494&532_256&537_260&457_498&452-0_0_21_32_3_32_29-125-23.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = [[254, 457, 497, 528]]
    keypoints = [(494, 532),
                 (256, 537),
                 (260, 457),
                 (498, 452), ]
    category_ids = [1]
    category_id_to_name = {1: 'PL'}
    # visualize(image, bboxes, keypoints, category_ids, category_id_to_name)

    class_labels = ['PL']

    transformed = transform(image=image, bboxes=bboxes, keypoints=keypoints, category_ids=category_ids)
    visualize(
        transformed['image'],
        transformed['bboxes'],
        transformed['keypoints'],
        transformed['category_ids'],
        category_id_to_name,
    )
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_keypoints = transformed['keypoints']
    transformed_class_labels = transformed['category_ids']
    if len(transformed_keypoints) < 4:
        print('---------')
    if len(transformed['bboxes']) < 1:
        print('数据增强导致目标越界，取消增强')

    plt.show()


if __name__ == '__main__':
    for i in range(5):
        main()
