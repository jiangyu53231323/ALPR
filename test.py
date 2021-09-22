import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.utils.data

from datasets.my_coco import COCO_eval
from datasets.yolo import YOLO_eval

from nets.hourglass import get_hourglass
from nets.resdcn import get_pose_net

from utils.utils import load_model
from utils.summary import create_logger
from utils.post_process import ctdet_decode

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='F:/code_download')
parser.add_argument('--log_name', type=str, default='coco_resdcn_18_384_dp')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='resdcn_18')

parser.add_argument('--img_size', type=int, default=384)

parser.add_argument('--test_flip', action='store_true')  # 控制是否有翻转的数据增强
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5

parser.add_argument('--test_topk', type=int, default=10)

parser.add_argument('--num_workers', type=int, default=4)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, 'checkpoint.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]


def main():
    logger = create_logger(save_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    max_per_image = 100

    Dataset_eval = COCO_eval if cfg.dataset == 'coco' else YOLO_eval
    dataset = Dataset_eval(cfg.data_dir, split='val', img_size=cfg.img_size,
                           test_scales=cfg.test_scales, test_flip=cfg.test_flip)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=True,
                                              collate_fn=dataset.collate_fn)

    print('Creating model...')
    if 'hourglass' in cfg.arch:
        model = get_hourglass[cfg.arch]
    elif 'resdcn' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=dataset.num_classes)
    else:
        raise NotImplementedError

    model = load_model(model, cfg.pretrain_dir)
    model = model.to(cfg.device)
    model.eval()

    results = {}
    with torch.no_grad():
        for inputs in tqdm(data_loader):
            img_id, inputs = inputs[0]

            detections = []
            for scale in inputs:  # 多个尺度如0.5,,0.75,1,1.25等
                inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
                image_scale = torch.from_numpy(inputs[scale]['image_scale'])
                padding_w = torch.from_numpy(inputs[scale]['padding_w'])
                padding_h = torch.from_numpy(inputs[scale]['padding_h'])

                output = model(inputs[scale]['image'])[-1]
                # output = [hmap, regs, w_h_]
                # 对模型的输出结果进行解码
                # dets-> [bboxes, scores, clses]
                dets = ctdet_decode(*output, [padding_w, padding_h], image_scale=image_scale, K=cfg.test_topk)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                top_preds = {}
                cls = dets[:, -1]
                for j in range(dataset.num_classes):
                    inds = (cls == j)
                    top_preds[j + 1] = dets[inds, 8:13].astype(np.float32)
                    top_preds[j + 1][:, :4] /= scale

                detections.append(top_preds)

            bbox_and_scores = {}
            for j in range(1, dataset.num_classes + 1):
                bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
            scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, dataset.num_classes + 1)])

            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, dataset.num_classes + 1):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            results[img_id] = bbox_and_scores

    eval_results = dataset.run_eval(results, cfg.ckpt_dir)
    print(eval_results)


if __name__ == '__main__':
    main()
