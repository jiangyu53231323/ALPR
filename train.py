import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from datasets.my_coco import COCO, COCO_eval
from datasets.yolo import YOLO, YOLO_eval

from nets.hourglass import get_hourglass
from nets.resdcn import get_pose_net

from utils.utils import _tranpose_and_gather_feature, load_model
from utils.image import transform_preds
from utils.my_losses import _heatmap_loss, _corner_loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctdet_decode

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=0)
# action=‘store_true’，只要运行时该变量有传参就将该变量设为True
parser.add_argument('--dist', action='store_true')  # 多GPU

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='F:/code_download')
parser.add_argument('--log_name', type=str, default='pascal_resdcn_18_384_dp')
parser.add_argument('--pretrain_name', type=str, default='pretrain')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'yolo'])
parser.add_argument('--arch', type=str, default='resdcn_18')

parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=1.25e-4)
parser.add_argument('--lr_step', type=str, default='45,90')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=170)

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

# logs目录
cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
# ckpt目录
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
# 预训练权重目录
cfg.pretrain_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_name, 'checkpoint.t7')

# 创建ckpt和logs目录
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)
# 学习率
cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    torch.manual_seed(317)  # 设置随机数种子
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training (OOM:Out Of Memory)

    num_gpus = torch.cuda.device_count()
    if cfg.dist:
        # 多GPU训练
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=num_gpus, rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')

    print('Setting up data...')

    dataset = COCO if cfg.dataset == 'coco' else YOLO
    train_dataset = dataset(cfg.data_dir, 'train', split_ratio=cfg.split_ratio, img_size=cfg.img_size)
    # 样本分发器，num_replicas为worker总数，rank为当前worker编号
    # 调用 train_dataset.__len__()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=num_gpus,
                                                                    rank=cfg.local_rank)
    # shuffle打乱数据集
    # sampler自定义从数据集中取样本的策略，如果指定则shuffle必须为False
    # pin_memory为True，data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中
    # drop_last为True，在最后一个不满batch_size的batch将会丢弃
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size // num_gpus
                                               if cfg.dist else cfg.batch_size,
                                               shuffle=not cfg.dist,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               sampler=train_sampler if cfg.dist else None)

    dataset_eval = COCO_eval if cfg.dataset == 'coco' else YOLO_eval
    val_dataset = dataset_eval(cfg.data_dir, 'val', test_scales=[1.], test_flip=False)
    # collate_fn 将一个list的sample组成一个mini-batch的函数
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    # 网络模型建立
    print('Creating model...')
    if 'hourglass' in cfg.arch:
        model = get_hourglass[cfg.arch]
    elif 'resdcn' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=train_dataset.num_classes)
    else:
        raise NotImplementedError

    # 多机多卡训练 DDP
    if cfg.dist:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cfg.device)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[cfg.local_rank, ],
                                                    output_device=cfg.local_rank)
    else:
        # DP
        model = nn.DataParallel(model).to(cfg.device)

    # 加载预训练权重
    if os.path.isfile(cfg.pretrain_dir):
        model = load_model(model, cfg.pretrain_dir)
    # 设置优化算法，lr衰减区间
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

    def train(epoch):
        print('\n Epoch: %d' % epoch)
        model.train()
        # perf_counter() 返回性能计数器的值（以分秒为单位），即具有最高可用分辨率的时钟，以测量短持续时间。
        # 返回值的参考点未定义，因此只有连续调用结果之间的差异有效
        tic = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                if k != 'meta':
                    # 数据送入GPU
                    batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

            outputs = model(batch['image'])

            '''------------------------------------------------------------'''
            # 得到 heat map, reg, wh 三个变量
            hmap, corner = zip(*outputs)
            # hmap = [h.permute(0, 2, 3, 1).contiguous() for h in hmap]  # from [bs c h w] to [bs, h, w, c]
            # corner = [c.permute(0, 2, 3, 1).contiguous() for c in corner]  # from [bs c h w] to [bs, h, w, c]
            # 分别计算 loss
            hmap_loss = _heatmap_loss(hmap, batch['hmap'])
            corner_loss = _corner_loss(corner, batch['corner'])
            # 进行 loss 加权，得到最终 loss
            loss = hmap_loss + 1 * corner_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                      ' hmap_loss= %.5f reg_loss= %.5f ' %
                      (hmap_loss.item(), corner_loss.item()) +
                      ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

                step = len(train_loader) * epoch + batch_idx
                summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
                summary_writer.add_scalar('corner_loss', corner_loss.item(), step)

        return

    def val_map(epoch):
        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()  # 释放cuda缓存
        max_per_image = 100

        results = {}
        with torch.no_grad():  # 不跟踪梯度，减少内存占用
            for inputs in val_loader:
                img_id, inputs = inputs[0]

                detections = []
                for scale in inputs:
                    inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
                    output = model(inputs[scale]['image'])[-1]
                    # 对检测结果进行后处理
                    dets = ctdet_decode(*output, K=cfg.test_topk)
                    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                    top_preds = {}
                    # x1,y1坐标做仿射变换
                    dets[:, :2] = transform_preds(dets[:, 0:2],
                                                  inputs[scale]['center'],
                                                  inputs[scale]['scale'],
                                                  (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    # x2,y2坐标做仿射变换
                    dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                                   inputs[scale]['center'],
                                                   inputs[scale]['scale'],
                                                   (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    # 合并同类
                    clses = dets[:, -1]
                    for j in range(val_dataset.num_classes):
                        inds = (clses == j)
                        top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                        top_preds[j + 1][:, :4] /= scale  # 恢复缩放

                    detections.append(top_preds)

                bbox_and_scores = {j: np.concatenate([d[j] for d in detections], axis=0)
                                   for j in range(1, val_dataset.num_classes + 1)}
                scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, val_dataset.num_classes + 1)])
                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, val_dataset.num_classes + 1):
                        keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                        bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

                results[img_id] = bbox_and_scores

        eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
        print(eval_results)
        summary_writer.add_scalar('val_mAP/mAP', eval_results[0], epoch)

    print('Starting training...')
    for epoch in range(1, cfg.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            val_map(epoch)
        print(saver.save(model.module.state_dict(), 'checkpoint'))
        lr_scheduler.step()  # move to here after pytorch1.1.0

    summary_writer.close()


if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()