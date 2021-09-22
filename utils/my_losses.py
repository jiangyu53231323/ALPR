import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss_slow(preds, targets):
    pos_inds = targets == 1  # todo targets > 1-epsilon ?
    neg_inds = targets < 1  # todo targets < 1-epsilon ?

    neg_weights = torch.pow(1 - targets[neg_inds], 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


# heatmap loss
def _heatmap_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    # gt()大于  ne()不等于  lt()小于  eq()等于
    pos_inds = targets.eq(1).float()  # heatmap 为 1 的部分是正样本
    neg_inds = targets.lt(1).float()  # 其他部分为负样本
    # pow(x,y) 方法返回 x^y（x的y次方） 的值
    neg_weights = torch.pow(1 - targets, 4)  # 对应 (1-Yxyc)^4

    loss = 0
    for pred in preds:  # 预测值
        # 约束在 0-1 之间
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds  # 正样本损失值
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds  # 负样本损失值

        num_pos = pos_inds.float().sum()  # 一个batch中的目标个数
        pos_loss = pos_loss.sum()  # 损失值累加
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss  # 只有负样本
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss / len(preds)  #


def _corner_loss(regs, gt_regs, mask):
    # expand_as 将输入tensor的维度扩展为与指定tensor相同的size
    # mask = mask[:, :, None].expand_as(gt_regs).float()
    '''
    将高斯分布内的像素点都参与到loss计算中，会导致corner_loss变得异常的大，如能达到25959。
    还是需要gaussian_mask来对loss大小进行限制，可以计算mask中的参加计算的像素点有多少，再将总的loss值除以总共的像素点数。
    regs[b,c,h,w]
    gt_regs[b,c,h,w]
    mask[b,c,h,w]
    '''
    num = copy.deepcopy(mask)
    num[num > 0] = 1
    batch, cat, height, width = mask.size()
    # topk_scores, topk_inds = torch.topk(mask.view(batch, cat, -1), num.sum().int() // batch)
    loss = 0
    # 单独计算每一个图片的loss
    for b in range(batch):
        # 按照heat-map值的大小顺序，获取当前图片的mask
        topk_score, topk_ind = torch.topk(mask[b].view(cat, -1), num[b].sum().int())
        topk_ind = topk_ind.expand(8, topk_ind.size(1))  # 将mask的维度调整到与reg相同
        reg = regs[0][b].view(8, -1).gather(1, topk_ind)  # 沿给定轴，将索引向量mask中值在reg上进行聚合
        gt_reg = gt_regs[b].view(8, -1).gather(1, topk_ind)
        # 计算L1损失的值，除以参与计算的总点数
        loss = loss + (F.l1_loss(reg, gt_reg, reduction='sum') / (num[b].sum() + 1e-4))
    # reg = regs[0] * mask
    # loss = F.l1_loss(reg, gt_regs, reduction='sum') / (mask.sum() + 1e-4)
    return loss/batch


def _w_h_loss(regs, gt_regs, mask):
    # expand_as 将输入tensor的维度扩展为与指定tensor相同的size
    # mask = mask[:, :, None].expand_as(gt_regs).float()
    '''
    同_corner_loss，此方法计算包围盒的loss，可使用的方法有L1损失、IoU、Giou、Diou、Ciou
    regs[b,c,h,w]
    gt_regs[b,c,h,w]
    mask[b,c,h,w]
    '''
    num = copy.deepcopy(mask)
    num[num > 0] = 1
    batch, cat, height, width = mask.size()
    # topk_scores, topk_inds = torch.topk(mask.view(batch, cat, -1), num.sum().int() // batch)
    loss = 0
    for b in range(batch):
        topk_score, topk_ind = torch.topk(mask[b].view(cat, -1), num[b].sum().int())
        topk_ind = topk_ind.expand(4, topk_ind.size(1))
        reg = regs[0][b].view(4, -1).gather(1, topk_ind)
        gt_reg = gt_regs[b].view(4, -1).gather(1, topk_ind)
        loss = loss + (F.l1_loss(reg, gt_reg, reduction='sum') / (num[b].sum() + 1e-4))
    # reg = regs[0] * mask
    # loss = F.l1_loss(reg, gt_regs, reduction='sum') / (mask.sum() + 1e-4)
    return loss/batch
