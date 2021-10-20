import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

class ComputeLoss:
    '''
    Loss calculation
    '''
    def __init__(self, model):
        super().__init__()
        self.device = model.device
        self.cfg = model.cfg

        # Define criteria
        self.cls_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.obj_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.degree_loss = nn.BCEWithLogitsLoss().to(self.device)

        det = model.model[-1]  # Detect() module
        self.balance = self.cfg['tobj_balance'] # P3-P7; tobj balance for different universe
        for k in 'na', 'nc', 'nl', 'anchors', 'n_degree':
            setattr(self, k, getattr(det, k))

    def __call__(self, pred, targets):  # predictions, targets, model
        '''
        pred: len(pred) == 3
              pred[0].shape = [batch size, num of anchors, num of grids h, num of grids w, 5 + n_classes + n_degree]
                               5 refers to cx, cy, cw, ch, conf.
        targets: shape = [num of bboxes, 7]
                         7 features: batch index, cls id, cx, cy, cw, ch, degree; ratio value.
        '''
        loss_cls = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)
        loss_obj = torch.zeros(1, device=self.device)
        loss_degree = torch.zeros(1, device=self.device)
        t_cls, t_bbox, t_indices, t_anchor, t_degree = self.build_targets(pred, targets)# targets

        # Losses
        for i_universe in range(len(pred)):  # layer index, layer predictions
            i_batch, i_anchor, grid_iy, grid_ix = t_indices[i_universe] # ATTENTION: grid_iy indicates the ith grid on y axis (height direction)
            tobj = torch.zeros_like(pred[i_universe][..., 4], device=self.device)  # target obj
            n_targets = len(i_batch)  # number of targets
            if n_targets > 0:
                pred_res = pred[i_universe][i_batch, i_anchor, grid_iy, grid_ix]  # prediction subset corresponding to targets
                assert len(pred_res) == n_targets, 'len(pred_res) != n_targets'
                # Regression
                # the predicted center could locate in the 8 neighbors (grids) of the target grid
                pred_xy = pred_res[:, :2].sigmoid() * 2. - 0.5
                pred_wh = (pred_res[:, 2:4].sigmoid() * 2) ** 2 * t_anchor[i_universe] # 0~4 * anchor size
                pred_bbox = torch.cat((pred_xy, pred_wh), 1)  # predicted box

                # all values here are in grid unit
                iou = self.calc_ciou(pred_bbox, t_bbox[i_universe])
                loss_box += (1.0 - iou).mean()  # iou loss
                # Objectness
                tobj[i_batch, i_anchor, grid_iy, grid_ix] = iou.detach().clamp(0).type(tobj.dtype)
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.zeros_like(pred_res[:, 5:5+self.nc], device=self.device)  # targets
                    t[range(n_targets), t_cls[i_universe]] = 1
                    loss_cls += self.cls_loss(pred_res[:, 5:5+self.nc], t)  # BCE
                # Degree
                t_degrees = torch.zeros_like(pred_res[:, 5 + self.nc:])
                if self.cfg['gaussian_degree']:
                    for i_degree in range(n_targets):
                        degree = t_degree[i_universe][i_degree]
                        t_degrees[i_degree] = self.gaussian_label(degree)
                else:
                    t_degrees[range(n_targets), t_degree[i_universe]] = 1
                loss_degree += self.degree_loss(pred_res[:, 5 + self.nc:], t_degrees)
            # if n_targets is 0, tobj is all zero
            loss_obj += self.obj_loss(pred[i_universe][..., 4], tobj) * self.balance[i_universe]

        loss_box *= self.cfg['loss_weight']['box']
        loss_obj *= self.cfg['loss_weight']['obj']
        loss_cls *= self.cfg['loss_weight']['cls']
        loss_degree *= self.cfg['loss_weight']['degree']

        loss = loss_box + loss_obj + loss_cls + loss_degree
        res = {'loss': loss,
               'loss_box': loss_box,
               'loss_obj': loss_obj,
               'loss_cls': loss_cls,
               'loss_degree': loss_degree}
        return res

    def build_targets(self, pred, targets):
        '''
        From targets to values to be used in loss calculation
        Distance here is measured in grid unit (how many grids does it occupy),
        so we need to change the targets cx,cy, cw,ch from absolute to grid unit.
        pred: multi-channel outputs
        targets: shape = [num of bboxes, 7]
                         7 features: batch index, cls id, cx, cy, cw, ch, degree; ratio value.
        ATTENTION: targets has no batch size dim.
                   For different universe targets doesn't change anyway.
        '''
        t_cls, t_bbox, t_indices, t_anchor, t_degree = [], [], [], [], []
        gain = torch.ones(targets.shape[1] + 1, device=self.device)  # normalized to gridspace gain
        # enlarge targets with anchor index
        # e.g. targets from
        #  [[batch id, cls id, cx, cy, cw, ch, degree]]; shape: (1, 7)
        # to
        # [[[batch id, cls id, cx, cy, cw, ch, degree, 0]],
        #  [[batch id, cls id, cx, cy, cw, ch, degree, 1]],
        #  [[batch id, cls id, cx, cy, cw, ch, degree, 2]]]; shape: (3, 1, 8)
        targets = torch.row_stack([torch.column_stack((targets,
                                                       torch.ones(targets.shape[0], 1, device=self.device) * i))
                                   for i in range(self.na)]).reshape(self.na, -1, targets.shape[1] + 1)
        for i_universe in range(len(pred)): # three output layers, three universes
            anchors = self.anchors[i_universe] # grid unit
            # gain for cx, cy, cw, ch are num of grids on certain axis in current universe
            gain[2:6] = torch.tensor(pred[i_universe].shape)[[3, 2, 3, 2]]
            # Match targets to anchors
            # self.anchors have been transferred into grid unit in Detect() already
            # Now we change targets into the same scale.
            # From now on, everything is counted in grid unit.
            t = targets * gain
            if len(targets) > 0:
                # If a robndbox is too large or too small compared with anchors, ignore it.
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                keep = torch.logical_and(torch.max(r, 2)[0] < self.cfg['anchor_t'],
                                         torch.min(r, 2)[0] > 1 / self.cfg['anchor_t'])
                t = t[keep]  # filter
                # ATTENTION: t[:, 2] is cx so cx.long() is the ith grid on the x axis(width direction)
                grid_ix_iy = deepcopy(t[:, 2:4].long())
                # Find two more neighbor grids
                for i in range(len(t)):
                    for delta in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        new_target = deepcopy(t[i])
                        neighbor_xy = new_target[2:4] + torch.tensor(delta, device=self.device).float() * self.cfg['offset']
                        if (0 <= neighbor_xy[0] < gain[2] and neighbor_xy[0].long() != t[i][2].long()) or\
                           (0 <= neighbor_xy[1] < gain[3] and neighbor_xy[1].long() != t[i][3].long()):
                            t = torch.cat([t, new_target[None, ...]])
                            grid_ix_iy = torch.cat([grid_ix_iy, neighbor_xy.long()[None, ...]])
                        else:
                            del new_target
            else:
                # If targets is empty, it should be in shape of (3, 0, 8)
                # Then we need targets[0] which is in shape of (0, 8) as the original one
                t = targets[0]

            i_batch = t[:, 0].long()
            grid_ix = grid_ix_iy[:, 0]
            grid_iy = grid_ix_iy[:, 1]
            i_anchor = t[:, -1].long()
            t_indices.append((i_batch, i_anchor, grid_iy, grid_ix))
            t_bbox.append(torch.cat((t[:, 2:4] - grid_ix_iy, t[:, 4:6]), 1)) # (offset_x, offset_y, cw, ch) all in grid unit
            t_anchor.append(anchors[i_anchor])
            t_cls.append(t[:, 1].long())
            # the granularity is based on n_degree
            t_degree.append((t[:, -2] * self.n_degree / 90).long())

        return t_cls, t_bbox, t_indices, t_anchor, t_degree

    def gaussian_label(self, degree):
        '''
        Borrowed from github: https://github.com/BossZard/rotation-yolov5
        It's like one-hot transform.
        degree: predicted angle degree
        mean, std: mean and standard deviation values for gaussian distribution.
        '''
        n_degree = torch.tensor(self.n_degree, device=self.device)
        x = torch.arange(torch.floor(- n_degree / 2), torch.ceil(n_degree / 2))
        dist = torch.exp(-(x - self.cfg['gaussian_smooth']['mean']) ** 2 / (2 * self.cfg['gaussian_smooth']['std'] ** 2))
        return torch.cat([dist[torch.ceil(n_degree / 2).long() - degree:],
                          dist[:torch.ceil(n_degree / 2).long() - degree]])

    @staticmethod
    def calc_ciou(box0, box1, eps=1e-7):
        '''
        PyTorch
        CIOU calculation
        box0, box1: shape could be [4,] or [num_target, 4]
                    cx,cy,cw,ch could be abs values or percentage values.
        Ref: https://arxiv.org/abs/1911.08287v1
             https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        '''
        # To do the transpose in case of contiguous problems
        box0 = box0.T
        box1 = box1.T
        b0_cx, b0_cy, b0_cw, b0_ch = box0
        b1_cx, b1_cy, b1_cw, b1_ch = box1
        b0_xmin, b0_xmax = b0_cx - b0_cw / 2, b0_cx + b0_cw / 2
        b0_ymin, b0_ymax = b0_cy - b0_ch / 2, b0_cy + b0_ch / 2
        b1_xmin, b1_xmax = b1_cx - b1_cw / 2, b1_cx + b1_cw / 2
        b1_ymin, b1_ymax = b1_cy - b1_ch / 2, b1_cy + b1_ch / 2
        # Intersection area
        inter = (torch.min(b0_xmax, b1_xmax) - torch.max(b0_xmin, b1_xmin)).clamp(0) * \
                (torch.min(b0_ymax, b1_ymax) - torch.max(b0_ymin, b1_ymin)).clamp(0)
        # Union Area
        union = b0_cw * b0_ch + b1_cw * b1_ch - inter
        iou = inter / (union + eps)

        cw = torch.max(b0_xmax, b1_xmax) - torch.min(b0_xmin, b1_xmin) # convex (smallest enclosing box) width
        ch = torch.max(b0_ymax, b1_ymax) - torch.min(b0_ymin, b1_ymin) # convex height
        c_square = cw ** 2 + ch ** 2 # convex diagonal squared
        rho_square = (b0_cx - b1_cx) ** 2 + (b0_cy - b1_cy) ** 2
        v = (4 / np.pi ** 2) * torch.pow(torch.atan(b0_cw / b0_ch) - torch.atan(b1_cw / b1_ch), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho_square / (c_square + eps) + v * alpha)