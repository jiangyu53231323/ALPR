import torch
import torch.nn as nn
from collections import OrderedDict


def _gather_feature(feat, ind, mask=None):
    # feat : [bs, wxh, c]
    dim = feat.size(2)
    # ind : [bs, index, c]
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # 沿给定轴dim，将输入索引张量index指定位置的值进行聚合 torch.gather(input, dim, index, out=None)
    feat = feat.gather(1, ind)  # 按照 dim=1 获取 ind
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feature(feat, ind):
    # ind 代表的是 ground truth 中设置的正样本的索引
    feat = feat.permute(0, 2, 3, 1).contiguous()  # from [bs c h w] to [bs, h, w, c]
    feat = feat.view(feat.size(0), -1, feat.size(3))  # to [bs, wxh, c]
    feat = _gather_feature(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2,
                      tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def load_model(model, pretrain_dir):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()  # 对字典对象中的元素排序

    # convert data_parallal to model 去掉module字符
    for key in state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]

    # check loaded parameters and created model parameters  去掉module字符
    model_state_dict_ = model.state_dict()
    model_state_dict = OrderedDict()
    for key in model_state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            model_state_dict[key[7:]] = model_state_dict_[key]
        else:
            model_state_dict[key] = model_state_dict_[key]
    # 检查权重格式
    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape != model_state_dict[key].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model


def count_parameters(model):
    num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if 'aux' not in k]
    print("Total num of param = %f M" % sum(num_paras))


def count_flops(model, input_size=384):
    flops = []
    handles = []

    def conv_hook(self, input, output):
        flops.append(output.shape[2] ** 2 *
                     self.kernel_size[0] ** 2 *
                     self.in_channels *
                     self.out_channels /
                     self.groups / 1e6)

    def fc_hook(self, input, output):
        flops.append(self.in_features * self.out_features / 1e6)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(fc_hook))

    with torch.no_grad():
        _ = model(torch.randn(1, 3, input_size, input_size))
    print("Total FLOPs = %f M" % sum(flops))

    for h in handles:
        h.remove()
