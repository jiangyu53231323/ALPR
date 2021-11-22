import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.my_losses import cross_entropy_loss


def cross_entropy_loss():
    x_input = torch.randn(3, 3)  # 随机生成输入
    print('x_input:\n', x_input)
    y_target = torch.tensor([1, 2, 0])  # 设置输出具体值 print('y_target\n',y_target)

    # 计算输入softmax，此时可以看到每一行加到一起结果都是1
    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(x_input)
    print('soft_output:\n', soft_output)

    # 在softmax的基础上取log
    log_output = torch.log(soft_output)
    print('log_output:\n', log_output)

    # 对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
    logsoftmax_func = nn.LogSoftmax(dim=1)
    logsoftmax_output = logsoftmax_func(x_input)
    print('logsoftmax_output:\n', logsoftmax_output)

    # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
    nllloss_func = nn.NLLLoss()
    nlloss_output = nllloss_func(logsoftmax_output, y_target)
    print('nlloss_output:\n', nlloss_output)

    # 直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
    crossentropyloss = nn.CrossEntropyLoss()
    crossentropyloss_output = crossentropyloss(x_input, y_target)
    print('crossentropyloss_output:\n', crossentropyloss_output)


def unify_loss(pre, target, cfg):
    loss = 0.0
    batch = pre[0].size()[0]
    # cls_score, cls_ind = torch.topk(target[0].view(batch, -1))
    for b in range(batch):
        # blue车牌
        if target['labels_class'][b] == 0:
            for j in range(7):
                l = target[b, j].to(cfg.device).long()
                p = pre[1][j][b].unsqueeze(0)
                loss += cross_entropy_loss(p, l)

        # green车牌
        else:
            for j in range(8):
                l = target[b, j].to(cfg.device).long()
                p = pre[2][j][b].unsqueeze(0)
                loss += cross_entropy_loss(p, l)
    loss += cross_entropy_loss(pre[0], target['labels_class'].to(cfg.device).long(), label_smooth=0.05)

    # for j in range(7):
    #     l = target[:, j].to('cuda:0').long()
    #     loss += F.cross_entropy(pre[1][j], l)  # 交叉熵损失函数
    return loss


def scr_decoder(pre, target):
    for k in target:
        target[k] = target[k].to('cpu')
    cls = pre[0].to('cpu')
    topk_score, topk_ind = torch.topk(cls, 1)
    num = 0
    for b in range(pre[0].size()[0]):
        # blue车牌检测
        if topk_ind[b] == 0:
            for k in range(7):
                p = pre[1][k][b].topk(1)
                if p == target['labels'][b][k]:
                    isTure = 1
                    continue
                else:
                    isTure = 0
                    break
            if isTure == 0:
                continue
            else:
                num = num + 1
        # green车牌检测
        else:
            for k in range(8):
                p = pre[2][k][b].topk(1)
                if p == target['labels'][b][k]:
                    isTure = 1
                    continue
                else:
                    isTure = 0
                    break
            if isTure == 0:
                continue
            else:
                num = num + 1
    return num

