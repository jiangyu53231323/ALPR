import os
import sys
from time import time
import argparse
from torch.utils.data import Dataset
# from imutils import paths
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from nets.scrnet import SCRNet
import pycocotools.coco as coco

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def get_image_path(image_dir, image_name):
    path_list = os.listdir(image_dir)
    p = ''
    for p in path_list:
        img_dir = os.path.join(image_dir, p)
        img_path = os.path.join(img_dir, image_name)
        if os.path.exists(img_path):
            p = img_path
            return img_path
    assert p != ''


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


def trainandsave():
    global min_loss
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                 "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
    args = {'input': 'C:\\data'}
    model_path = './models/net_params9.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    imgSize = (192, 64)
    batchSize = 96

    train_dataset = DataLoader(args["input"], imgSize, 'train')
    eval_dataset = DataLoader(args["input"], imgSize, 'eval')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batchSize,
                                               shuffle=True,
                                               num_workers=6,
                                               pin_memory=True,
                                               drop_last=True,
                                               )
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True,
                                              drop_last=True,
                                              collate_fn=eval_dataset.collate_fn
                                              )

    # 神经网络结构
    model = SCRNet()
    # net.load_state_dict(torch.load(model_path))  # 加载已训练好的参数
    model.to(device)
    model.train()
    # 设置优化算法，lr衰减区间
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)

    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分
    start = time()
    min_running_loss = 100
    for epoch in range(30):  # 训练的数据量为50个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出

        for i, data in enumerate(train_loader, 0):  # 传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            # Y0 = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
            # Y = one_hot(Y0, batchSize)  # 标签转换成one-hot

            # wrap them in Variable
            inputs = inputs.to(device)  # 转换数据格式用Variable
            # labels = labels.to(device)
            # Y = Variable(Y)

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = model(inputs)  # 把数据输进CNN网络net

            loss = 0.0
            # loss = criterion(outputs, labels)  # 计算损失值
            for j in range(7):
                # l = Variable(torch.LongTensor([el[j] for el in labels]))
                l = torch.LongTensor(labels[j]).to(device)
                o = outputs[1][j]
                loss += criterion(outputs[1][j], l)
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.item()  # loss累加

            if i % 100 == 99:
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 100))  # 然后再除以200，就得到这两百次的平均损失值
                if (running_loss / 100) < min_running_loss:
                    min_running_loss = (running_loss / 100)
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
        lr_scheduler.step()
    end = time()
    print(end - start)
    print('Finished Training')
    # 保存神经网络
    torch.save(obj=model.state_dict(), f='models/net_params_v1_1.pth')  # 保存整个神经网络的结构和模型参数
    # if min_running_loss < min_loss:
    #     print("保存参数")
    #     min_loss = min_running_loss
    #     torch.save(obj=model.state_dict(), f='models/net_params_v1_1.pth')  # 只保存神经网络的模型参数


def main():
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(317)  # 设置随机数种子


if __name__ == '__main__':
    trainandsave()
