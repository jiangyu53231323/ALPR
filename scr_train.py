import copy
import os
import sys
import time
import argparse

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from nets.ghostnet import My_GhostNet
from nets.mobilenet import MobileNetV3_Small, MobileNetV3_Large
from nets.scrnet import SCRNet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from datasets.my_coco import COCO, COCO_eval
from datasets.scr_coco import SCR_COCO, SCR_COCO_eval
from datasets.yolo import YOLO, YOLO_eval

from nets.hourglass import get_hourglass
# from nets.resdcn import get_pose_net
# from nets.resdcn_cbam import get_pose_net
from nets.resdcn_cbam_fpn import get_pose_net

from utils.utils import _tranpose_and_gather_feature, load_model
from utils.image import transform_preds
from utils.my_losses import _heatmap_loss, _corner_loss, _w_h_loss, bboxes_loss, scr_loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctdet_decode
from datasets.CudaDataLoader import CudaDataLoader, MultiEpochsDataLoader

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=0)
# action=‘store_true’，只要运行时该变量有传参就将该变量设为True，即触发时为True不触发为false
parser.add_argument('--dist', action='store_true')  # 多GPU

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='E:\CodeDownload\data')
parser.add_argument('--log_name', type=str, default='scr_coco_ml_64x224_se_fpn')
parser.add_argument('--pretrain_name', type=str, default='scr_pretrain')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'yolo'])
parser.add_argument('--arch', type=str, default='scrnet')

parser.add_argument('--img_size', type=int, default=(224, 64))  # 长×宽
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=1.25e-4)
parser.add_argument('--lr_step', type=str, default='2,4,6')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=20)

parser.add_argument('--test_topk', type=int, default=10)

parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--val_interval', type=int, default=2)
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

    dataset = SCR_COCO if cfg.dataset == 'coco' else YOLO
    train_dataset = dataset(cfg.data_dir, cfg.img_size, 'train')
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
                                               sampler=train_sampler if cfg.dist else None
                                               )
    dataset_eval = SCR_COCO_eval if cfg.dataset == 'coco' else YOLO_eval
    val_dataset = dataset_eval(cfg.data_dir, cfg.img_size, 'val')
    # collate_fn 将一个list的sample组成一个mini-batch的函数
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size // num_gpus
                                             if cfg.dist else cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True, )
    # 网络模型建立
    print('Creating model...')
    if 'scrnet' in cfg.arch:
        model = SCRNet()
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
    # 损失函数
    crossentropyloss = nn.CrossEntropyLoss()

    def train(epoch):
        print('\n Epoch: %d' % epoch)
        model.train()
        print(' learning rate: %e' % optimizer.param_groups[0]['lr'])
        # perf_counter() 返回性能计数器的值（以分秒为单位），即具有最高可用分辨率的时钟，以测量短持续时间。
        # 返回值的参考点未定义，因此只有连续调用结果之间的差异有效
        tic = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            # for k in batch:
            #     # if k == 'labels':
            #     #     batch[k] = torch.LongTensor(batch[k]).to(cfg.device)
            #     if k == 'image':
            #         # 数据送入GPU
            #         batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

            outputs = model(batch['image'].to(device=cfg.device, non_blocking=True))

            # loss = 0.0
            # for j in range(7):
            #     l = batch['labels'][:, j].to(cfg.device).long()
            #     # l = l.long()
            #     loss += crossentropyloss(outputs[1][j], l)  # 交叉熵损失函数
            province_loss = crossentropyloss(outputs[0], batch['labels'][:, 0].to(cfg.device).long())
            ctc_loss = scr_loss(outputs, batch['labels'], batch['labels_size'], cfg)
            loss = province_loss + ctc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                      ' loss= %.5f' % loss.item() + ' (%d samples/sec)' % (
                              cfg.batch_size * cfg.log_interval / duration))

                step = len(train_loader) * epoch + batch_idx
                summary_writer.add_scalar('loss', loss.item(), step)

        return

    def val(epoch):
        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()  # 释放cuda缓存
        max_per_image = 100
        amount = val_loader.dataset.num_samples
        results = {}
        num = 0
        with torch.no_grad():  # 不跟踪梯度，减少内存占用
            for i, inputs in enumerate(val_loader):
                # img_id, inputs = inputs[0]
                inputs['image'] = inputs['image'].to(cfg.device)
                inputs['labels'] = inputs['labels'].to(cfg.device)
                outputs = model(inputs['image'])
                out = [torch.topk(ee, 1)[1].squeeze(1) for ee in outputs[1]]
                isTure = 1
                for b in range(len(inputs['labels'])):
                    for j in range(7):
                        if inputs['labels'][b][j] == out[j][b]:
                            isTure = 1
                            continue
                        else:
                            isTure = 0
                            break
                    if isTure == 0:
                        continue
                    else:
                        num = num + 1

        accuracy = float(num) / float(amount)
        print('amount = %d' % amount)
        print('accuracy = %f' % accuracy)
        summary_writer.add_scalar('val_mAP/mAP', accuracy, epoch)

    print('Starting training...')
    for epoch in range(1, cfg.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            val(epoch)
        print(saver.save(model.module.state_dict(), 'checkpoint'))
        lr_scheduler.step()  # move to here after pytorch1.1.0

    summary_writer.close()


if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
