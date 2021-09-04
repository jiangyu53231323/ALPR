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


def ctdet_decode(hmap, regs, w_h_, K=100):
    '''
    hmap 提取中心点位置为 xs,ys
    regs 保存的是偏置，需要加在 xs,ys 上，代表精确的中心位置
    w_h_ 保存的是对应目标的宽和高
    '''
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:  # batch > 1 代表使用了翻转
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]

    batch = 1
    # 这里的 nms 和带 anchor 的目标检测方法中的不一样，这里使用的是 3x3 的 maxpool 筛选
    hmap = _nms(hmap)  # perform nms on heatmaps
    # 找到前 K 个极大值点代表存在目标
    scores, inds, clses, ys, xs = _topk(hmap, K=K)
    # 找到前K个极大值点对应的偏置值
    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    # 中心点坐标 = 中心点像素位置 + 偏移
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]
    # 找到前K个极大值点对应的宽高
    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    # xs,ys 是中心坐标，w_h_[...,0:1] 是 w,1:2 是 h
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections
