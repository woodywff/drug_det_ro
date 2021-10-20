
import torch
import torch.nn as nn
import math
from tools.utils import make_divisible
from copy import deepcopy
from pt.common import Conv, Bottleneck, SPP, Focus, C3, Concat

import numpy as np
from tools.const import DEBUG_FLAG


class Detect(nn.Module):
    def __init__(self, nc, anchors=(), ch=(), n_degree=90):  # detection layer
        '''
        nc: number of classes
        anchors: refer to yolov5s_ro.yml
        ch: input channel; e.g. [128, 256, 512]
        n_degree: number of rotate degree
        '''
        super().__init__()
        self.nc = nc
        self.n_degree = n_degree
        self.no = 5 + nc + n_degree  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers; num of universes
        self.na = len(anchors[0]) // 2  # number of anchors (in each universe)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # x shape: (bs,3*(5+n_classes+n_degree),20,20) to x shape: (bs,3,20,20,5+n_classes+n_degree)
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x


class RoYolov5(nn.Module):
    def __init__(self, cfg, nc, infer_only=False):
        '''
        cfg: model config .yaml file
        nc: number of classes
        infer_only: if True, only for inference.
        '''
        super().__init__()
        self.cfg = cfg
        self.n_out_layer = len(self.cfg['anchors'])
        self.nc = nc
        # Define model
        ch = self.cfg['ch']   # input channels
        self.model, self.save = self.parse_model(deepcopy(self.cfg), ch=[ch])  # model, savelist
        # Build strides, anchors
        self.det = self.model[-1]  # Detect()
        assert isinstance(self.det, Detect), 'The last layer should be Detect.'
        self.det.stride = self.stride = torch.tensor(self.cfg['stride'])
        self.det.anchors /= self.det.stride.view(-1, 1, 1)
        self.grid = [torch.zeros(1)] * self.n_out_layer  # init grid
        if not infer_only:
            self.check_anchor_order(self.det)
            # Init weights, biases
            self._initialize_weights()
            self._initialize_biases()  # only run once

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        '''
        RETURN: All values are measured in grid unit.
        '''
        y, dt = [], []  # outputs
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            if DEBUG_FLAG:
                with open(f'present/layer_{i}.txt', 'w') as f:
                    f.write(str(m))
                if isinstance(x, list):
                    for j in range(len(x)):
                        np.save(f'present/out_{i}_{j}.npy', x[j].detach().cpu().numpy())
                else:
                    np.save(f'present/out_{i}.npy', x.detach().cpu().numpy())
            y.append(x if m.i in self.save else None)  # save output
        return x

    def post_process(self, pred):
        '''
        Specific for val and infer
        pred: len(pred) == 3
              pred[0].shape = [batch size, num of anchors, num of grids h, num of grids w, 5 + n_classes + n_degree]
                               5 refers to cx, cy, cw, ch, conf.
        '''
        res = []
        for i_universe in range(len(pred)):
            # ATTENTION: n_grid_y indicates the number of grids on y axis (height direction)
            batch_size, _, n_grid_y, n_grid_x, dim_out = pred[i_universe].shape
            # grids indicates the coordinates of each grid, so it looks like (x, y)
            grids = torch.stack(torch.meshgrid([torch.arange(n_grid_x),
                                                torch.arange(n_grid_y)]),
                                -1)[None, None, ..., [1, 0]].to(self.device) # ATTENTION: x,y index changed
            if DEBUG_FLAG:
                np.save(f'present/post_grids_{i_universe}.npy', grids.detach().cpu().numpy())
            y = pred[i_universe].sigmoid()
            y[..., 0:2] = (y[..., :2] * 2. - 0.5 + grids) / \
                          torch.tensor([n_grid_x, n_grid_y], device=self.device)  # cx, cy; ratio value
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.det.anchor_grid[i_universe] / \
                          (torch.tensor([n_grid_x, n_grid_y], device=self.device) * self.stride[i_universe])  # cw, ch; ratio value
            if DEBUG_FLAG:
                np.save(f'present/post_y_{i_universe}.npy', y.detach().cpu().numpy())
            # That's it, from now on we don't need to remember the grid info anymore, so we mix all the predicted bounding boxes.
            res.append(y.view(batch_size, -1, dim_out))
        return torch.cat(res, 1)

    def parse_model(self, d, ch):  
        '''
        Create models from config file, e.g. pt/configs/yolov5s.yaml
        '''
        anchors, gd, gw = d['anchors'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        nc = self.nc
        n_degree = d['n_degree']
        no = na * (5 + nc + n_degree)  # number of outputs = anchors * (classes + 5)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except:
                    pass
            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [Conv, Bottleneck, SPP, Focus, C3]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
                if m == C3:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            elif m is Detect:
                args.append([ch[x] for x in f])
                args.append(n_degree)
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = ch[f]
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    def _initialize_weights(self):
        for m in self.model.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True
                
    def _initialize_biases(self, cf=None):
        '''
        Initialize biases into Detect(), cf is class frequency
        https://arxiv.org/abs/1708.02002 section 3.3
        '''
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def check_anchor_order(self, m):
        '''
        Check anchor order against stride order for Detect() module m, and correct if necessary
        '''
        a = m.anchor_grid.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = m.stride[-1] - m.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            print('Reversing anchor order')
            m.anchors[:] = m.anchors.flip(0)
            m.anchor_grid[:] = m.anchor_grid.flip(0)
