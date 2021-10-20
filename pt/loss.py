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

        det = model.model[-1]  # Detect() module
        self.balance = self.cfg['balance'] # P3-P7
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, pred, targets):  # predictions, targets, model
        '''
        pred: len(pred) == 3
              pred[0].shape = [batch size, num of anchors, num of grids h, num of grids w, 85]
                               85 features: cx, cy, cw, ch, conf, 80 classes prop
                               cx, cy, cw, ch have no physical meanings at this moment.
        targets: shape = [num of bboxes, 6]
                         6 features: batch index, cls id, cx, cy, cw, ch
                         cx, cy, cw, ch are all percentage values with respect to the abs width and height.
        '''
        loss_cls = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)
        loss_obj = torch.zeros(1, device=self.device)
        t_cls, t_bbox, t_indices, t_anchor = self.build_targets(pred, targets)# targets

        # Losses
        for i in range(len(pred)):  # layer index, layer predictions
            i_batch, i_anchor, grid_iy, grid_ix = t_indices[i] # ATTENTION: grid_iy indicates the ith grid on y axis (height direction)
            tobj = torch.zeros_like(pred[i][..., 0], device=self.device)  # target obj
            n_targets = len(i_batch)  # number of targets
            if n_targets > 0:
                pred_res = pred[i][i_batch, i_anchor, grid_iy, grid_ix]  # prediction subset corresponding to targets
                # Regression
                # the predicted center could locate in the 8 neighbors (grids) of the target grid
                pred_xy = pred_res[:, :2].sigmoid() * 2. - 0.5
                pred_wh = (pred_res[:, 2:4].sigmoid() * 2) ** 2 * t_anchor[i] # 0~4 * anchor size
                pred_bbox = torch.cat((pred_xy, pred_wh), 1)  # predicted box

                # all values here are in grid unit
                iou = self.calc_ciou(pred_bbox, t_bbox[i])
                loss_box += (1.0 - iou).mean()  # iou loss
                # Objectness
                tobj[i_batch, i_anchor, grid_iy, grid_ix] = iou.detach().clamp(0).type(tobj.dtype)
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.zeros_like(pred_res[:, 5:], device=self.device)  # targets
                    t[range(n_targets), t_cls[i]] = 1
                    loss_cls += self.cls_loss(pred_res[:, 5:], t)  # BCE
            loss_obj += self.obj_loss(pred[i][..., 4], tobj) * self.balance[i]

        loss_box *= self.cfg['box']
        loss_obj *= self.cfg['obj']
        loss_cls *= self.cfg['cls']

        loss = loss_box + loss_obj + loss_cls
        res = {'loss': loss,
               'loss_box': loss_box,
               'loss_obj': loss_obj,
               'loss_cls': loss_cls}
        return res

    def build_targets(self, pred, targets):
        '''
        From targets to values to be used in loss calculation
        Distance here is measured in grid unit (how many grids does it occupy),
        so we need to change the targets cx,cy, cw,ch from absolute to grid unit.
        pred: multi-channel outputs
        targets: [batch size, num of obj, label], label: (index in a batch, class, cx, cy, cw, ch)
        '''
        t_cls, t_bbox, t_indices, t_anchor = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        targets = torch.row_stack([torch.column_stack((targets,
                                                       torch.ones(targets.shape[0], 1, device=self.device) * i))
                                   for i in range(self.na)]).reshape(self.na, -1, targets.shape[1] + 1)

        for i_universe in range(len(pred)): # three output layers, three universes
            anchors = self.anchors[i_universe]
            gain[2:6] = torch.tensor(pred[i_universe].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # self.anchors have been transferred into grid size in Detect() already
            # Now we change targets into the same scale.
            # From now on, everything is counted in grid unit.
            if len(targets) > 0:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                keep = torch.logical_and(torch.max(r, 2)[0] < self.cfg['anchor_t'],
                                         torch.min(r, 2)[0] > 1 / self.cfg['anchor_t'])
                t = t[keep]  # filter
                grid_ix_iy = deepcopy(t[:, 2:4].long()) # ATTENTION: t[:, 2] is cx so cx.long() is the ith grid on the x axis(width direction)
                for i in range(len(t)):
                    for delta in [(0,1), (0,-1), (1,0), (-1,0)]:
                        new_target = deepcopy(t[i])
                        neighbor_xy = new_target[2:4] + torch.tensor(delta, device=self.device).float() * self.cfg['offset']
                        if (0 <= neighbor_xy[0] < gain[2] and neighbor_xy[0].long() != t[i][2].long()) or\
                           (0 <= neighbor_xy[1] < gain[3] and neighbor_xy[1].long() != t[i][3].long()):
                            t = torch.cat([t, new_target[None, ...]])
                            grid_ix_iy = torch.cat([grid_ix_iy, neighbor_xy.long()[None, ...]])
                        else:
                            del new_target
            else:
                t = targets[0]

            i_batch = t[:, 0].long()
            cls = t[:, 1].long()
            grid_ix = grid_ix_iy[:, 0]
            grid_iy = grid_ix_iy[:, 1]
            i_anchor = t[:, -1].long()
            t_indices.append((i_batch, i_anchor, grid_iy, grid_ix))
            t_bbox.append(torch.cat((t[:, 2:4] - grid_ix_iy, t[:, 4:6]), 1)) # (offset_x, offset_y, cw, ch) all in grid unit
            t_anchor.append(anchors[i_anchor])
            t_cls.append(cls)

        return t_cls, t_bbox, t_indices, t_anchor

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

