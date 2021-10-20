import torch
import torch.nn as nn
import math
from tools.utils import make_divisible
from copy import deepcopy
from pt.common import Conv, Bottleneck, SPP, Focus, C3, Concat
import numpy as np


class Detect(nn.Module):
    def __init__(self, nc, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x


class Yolov5(nn.Module):
    def __init__(self, cfg, nc, infer_only=False):  # model, input channels, number of classes
        '''
        nc: num of classes
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
        s = 256  # 2x min stride
        self.det.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        self.det.anchors /= self.det.stride.view(-1, 1, 1)
        self.check_anchor_order(self.det)
        self.stride = self.det.stride
        self.grid = [torch.zeros(1)] * self.n_out_layer  # init grid

        if not infer_only:
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
            y.append(x if m.i in self.save else None)  # save output
        return x

    def post_process(self, pred):
        '''
        Specific for val and infer
        '''
        res = []
        for i in range(len(pred)):
            batch_size, _, n_grid_y, n_grid_x, dim_out = pred[i].shape # ATTENTION: n_grid_y indicates the number of grids on y axis (height direction)
            grids = torch.stack(torch.meshgrid([torch.arange(n_grid_x),
                                                torch.arange(n_grid_y)]),
                                -1)[None,None,...,[1,0]].to(self.device) # ATTENTION: x,y index changed
            y = pred[i].sigmoid()
            y[..., 0:2] = (y[..., :2] * 2. - 0.5 + grids) / \
                          torch.tensor([n_grid_x, n_grid_y], device=self.device)  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.det.anchor_grid[i] / \
                          torch.tensor([n_grid_x, n_grid_y], device=self.device) / self.stride[i]  # wh
            res.append(y.view(batch_size, -1, dim_out))
        return torch.cat(res, 1)

    def parse_model(self, d, ch):  
        '''
        Create models from config file, e.g. pt/configs/yolov5s.yaml
        '''
        anchors, gd, gw = d['anchors'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        nc = self.nc
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

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
