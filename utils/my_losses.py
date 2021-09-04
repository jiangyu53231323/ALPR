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
def _neg_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    # gt()大于  ne()不等于  lt()小于  eq()等于
    pos_inds = targets.eq(1).float()  # heatmap 为 1 的部分是正样本
    neg_inds = targets.lt(1).float()  # 其他部分为负样本
    # pow() 方法返回 xy（x的y次方） 的值
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


def _reg_loss(regs, gt_regs, mask):
    # expand_as 将输入tensor的维度扩展为与指定tensor相同的size
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)
