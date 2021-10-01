import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='C:\data\CCPD2019\ccpd', type=str)
    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    img_filenames = os.listdir(opt.dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(opt.dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    # 以RGB通道顺序输出均值和方差
    print(m[0][::-1])
    print(s[0][::-1])


if __name__ == '__main__':
    main()
