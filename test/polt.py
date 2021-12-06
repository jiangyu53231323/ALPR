import json
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as draw
import matplotlib
print(matplotlib.get_backend())
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def draw(file_name):
    # 使用pandas第三方库读取txt文件
    train_log = pd.read_csv(file_name)
    # 设置对应的坐标轴和具体的内容
    # _, ax1 = plt.subplots()
    #
    # ax1.set_title("category and count")
    # 使用的折线图的方式
    # ax1.plot(train_log["category"], train_log["count"], alpha=0.5)
    # bar方法使用的是柱状图的形式
    values = train_log.values
    count = [int(i[1]) for i in values]
    category = ["{}".format(i[0]) for i in values]
    plt.bar(range(len(category)), count, align='center', color='steelblue')
    plt.title('category and count')
    plt.ylabel('count')
    plt.xticks(range(len(category)), category)
    plt.ylim([0, 450])
    for x, y in enumerate(count):
        plt.text(x, y + 2, '%s' % round(y, 1), ha='center')
    # plt.legend()
    # plt.savefig("./log1.png")
    plt.show()

    # plt.bar(train_log["category"], train_log["count"])
    # ax1.set_xlabel('category')
    # ax1.set_ylabel('count')
    # plt.legend(loc='upper left')
    # plt.savefig("./log1.png")
    # plt.show()


# path = input("输入需要复制文件目录：")
# 获取总的目录
# path = "./"
# # 获取总目录下的文件夹
# fileNames = os.listdir(path)
# # 最后结果保存到txt文件中
# file_name = "./train.txt"
# # 用于统计样本总量
# sum = 0
# # 打开txt文档，首先将存在的类型写入：这里我们有两项分别为 类别-数量
# # 第一个参数表示打开的文件名称，第二个参数a+表示可读可写，r表示可读，w表示可写
# with open(file_name, "a+") as f:
#     f.writelines("category" + ',' + "count")
#     f.writelines("\n")
#     # 遍历刚刚得到的总目录下的文件夹
# for fileName in fileNames:
#     # patha为每个文件夹
#     patha = path + '/' + fileName
#     # 统计每个文件夹内的数量
#     count = len(os.listdir(patha))
#     sum = count + sum
#     # 再次打开，这次写入每个文件夹对应的结果：类别-数量
#     with open(file_name, "a+") as f:
#         f.writelines(str(fileName) + ',' + str(count) + '\n')
#         # f.writelines("\n")
# print(sum)


# 将绘制柱状图抽取为单独的方法，参数为txt所在的地址。
file_name = "./train.txt"
draw(file_name)
