import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


# 使用3×3最大池化层代替nms
def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()  # 找到极大值点
    return heat * keep


# 寻找前K个极大值点
def _topk(scores, K=40):
    # score shape : [batch, class , h, w]
    batch, cat, height, width = scores.size()
    # to shape: [batch , class, h * w] 分类别，每个 class channel 统计最大值
    # topk_scores 和 topk_inds 分别是前 K 个 score 和对应的 下标序号
    # 此时的topk_scores为[batch,class,K]  topk_inds为[batch,class,K]
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    # 找到横纵坐标
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    # to shape: [batch , class * h * w] 这样的结果是不分类别的，全体 class 中最大的100个
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # 所有类别中找到最大值
    topk_clses = (topk_ind / K).int()  # 第一类就为0，第二类为1，以此类推
    topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(hmap, cors, bbs, padding, down_ratio=4, image_scale=1, K=100):
    '''
    hmap 提取中心点位置为 xs,ys
    cors 保存的是角点，是相对于中心点的坐标
    返回：corner(坐标形式)
         bboxes(坐标形式)
         scores 得分
         clses 类别
    '''
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # 这里的 nms 和带 anchor 的目标检测方法中的不一样，这里使用的是 3x3 的 maxpool 筛选
    hmap = _nms(hmap)  # perform nms on heatmaps
    # 找到前 K 个极大值点代表存在目标
    scores, inds, clses, ys, xs = _topk(hmap, K=K)
    # 找到前K个极大值点对应的偏置值
    cors = _tranpose_and_gather_feature(cors, inds)
    cors = cors.view(batch, K, 8)
    bbs = _tranpose_and_gather_feature(bbs, inds)
    bbs = bbs.view(batch, K, 4)

    '''四个角点的坐标，
    16是相对坐标缩放的系数，在 draw_corner_gaussian 中有相同数值使用
    down_ratio是特征图缩小的倍数
    padding是在my_image.py中resize_and_padding()后对图片的填充值，填充后会影响corner和bboxes的坐标
    image_scale是resize后图片与原始图片的比值，利用coco api计算iou是与原始图片的标注进行计算，而不是resize后的图片
    '''
    x1 = (xs.view(batch, K, 1) - cors[:, :, 0:1]) * down_ratio - (padding[0] // 2)
    y1 = (ys.view(batch, K, 1) - cors[:, :, 1:2]) * down_ratio - (padding[1] // 2)
    x2 = (xs.view(batch, K, 1) - cors[:, :, 2:3]) * down_ratio - (padding[0] // 2)
    y2 = (ys.view(batch, K, 1) + cors[:, :, 3:4]) * down_ratio - (padding[1] // 2)
    x3 = (xs.view(batch, K, 1) + cors[:, :, 4:5]) * down_ratio - (padding[0] // 2)
    y3 = (ys.view(batch, K, 1) + cors[:, :, 5:6]) * down_ratio - (padding[1] // 2)
    x4 = (xs.view(batch, K, 1) + cors[:, :, 6:7]) * down_ratio - (padding[0] // 2)
    y4 = (ys.view(batch, K, 1) - cors[:, :, 7:8]) * down_ratio - (padding[1] // 2)
    corners = torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], dim=2)
    corners = corners * image_scale
    # bboxes坐标
    bx1 = (xs.view(batch, K, 1) - bbs[:, :, 0:1]) * down_ratio - (padding[0] // 2)
    by1 = (ys.view(batch, K, 1) - bbs[:, :, 1:2]) * down_ratio - (padding[1] // 2)
    bx2 = (xs.view(batch, K, 1) + bbs[:, :, 2:3]) * down_ratio - (padding[0] // 2)
    by2 = (ys.view(batch, K, 1) + bbs[:, :, 3:4]) * down_ratio - (padding[1] // 2)
    bboxes = torch.cat([bx1, by1, bx2, by2], dim=2)
    bboxes = bboxes * image_scale

    clses = clses.view(batch, K, 1).float()  # [batch,k,1]
    scores = scores.view(batch, K, 1)  # [batch,k,1]

    detections = torch.cat([corners, bboxes, scores, clses], dim=2)  # detections[batch, K, corners+scores+clses]
    return detections
