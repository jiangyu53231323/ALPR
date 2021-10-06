import os
import time

import cv2

if __name__ == '__main__':
    img_dir = 'C:\data\CCPD2019\ccpd'
    save_dir = 'F:\code_download\data\CCPD_384'
    img_list = os.listdir(img_dir)
    tic = time.perf_counter()
    i = 0
    for img in img_list:
        i += 1
        img_path = os.path.join(img_dir, img)
        image = cv2.imread(img_path)
        # image = cv2.resize(image, (239, 384))
        save_path = os.path.join(save_dir, img)
        # cv2.imwrite(save_path, image)  # 写入图片
        if i % 1000 == 0:
            duration = time.perf_counter() - tic
            print(duration)
            break
    tic = time.perf_counter()
    for img in img_list:
        i += 1
        img_path = os.path.join(img_dir, img)
        image = cv2.imread(img_path)
        # image = cv2.resize(image, (720, 1160))
        save_path = os.path.join(save_dir, img)
        # cv2.imwrite(save_path, image)  # 写入图片
        if i % 1000 == 0:
            duration = time.perf_counter() - tic
            print(duration)
            break

