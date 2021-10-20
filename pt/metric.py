import pdb
import numpy as np
import time
import torch
import torchvision
from copy import deepcopy
from collections import Counter
from tools.plot import rotate_p_list as rotate_p_list_np
from pt.utils import rotate_p_list as rotate_p_list_pt
from math import pi

from tools.utils import json_read


class NMS:
    '''
    Non-Maximum Suppression
    '''
    def __init__(self, n_classes, is_ro=True, n_degree=90, conf_threshold=0.25, iou_threshold=0.45, hw_threshold=0.01,
                 patience=10, max_nms=30000, max_det=300, batched_nms=False, mask_h=640, mask_w=640,
                 extra_filter_record=None, extra_threshold=0.8):
        '''
        n_classes: number of obj classes
        is_ro: if True, for rotated bounding boxes.
        conf_threshold: threshold to filter out low trusted boxes; 0~1
        iou_threshold: for torchvision.ops.batched_nms, discards all overlapping boxes with IoU > iou_threshold; 0~1.
        hw_threshold: threshold of width and height, less than which do we ignore the box;0~1 ratio value.
        patience: waiting time before break.
        max_nms: maximum number of boxes into torchvision.ops.batched_nms()
        max_det: maximum number of kept boxes.
        batched_nms: if true, use batched_nms, otherwise nms.
        mask_h, mask_w: for is_ro=True; height and width of mask map for rotated iou calculation.
        extra_filter_record: specific size of certain objects
        extra_threshold: conf threshold less than which we will measure the size
        '''
        self.n_classes = n_classes
        self.is_ro = is_ro
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.hw_threshold = hw_threshold
        self.patience = patience
        self.max_nms = max_nms
        self.max_det = max_det
        self.batched_nms = batched_nms
        if is_ro:
            self.mask_h = mask_h
            self.mask_w = mask_w
            self.n_degree = n_degree
        # ATTENTION: Currently, we can deploy extra filter on bndbox
        # just because it only takes in account of four pills.
        self.extra_filter = ExtraFilter(extra_filter_record, extra_threshold) if extra_filter_record else None

    def __call__(self, pred):
        '''
        pred (torch.Tensor):
            output from post_process()
            shape = [batch size, num of predicted boxes, 5 + n_class + n_degree]
                     5 features: cx, cy, cw, ch, conf.
                     cx, cy, cw, ch are all percentage values with respect to the abs width and height.
        RETURN: (list of ndarray) len == batch size; shape = [num of predicted boxes after nms, 6 or 7]
                6 or 7 features: cx, cy, cw, ch, (degree), confidence, cls_id
                cx, cy, cw, ch are all ratio values with respect to the abs width and height.
        '''
        t = time.time()
        # init res with 0 values rather than []
        res = [0] * len(pred)
        for batch_i, x in enumerate(pred):  # image index, image inference
            if self.is_ro:
                x = x[(x[:, 4] > self.conf_threshold) & (x[:, 2] > self.hw_threshold) & (x[:, 3] > self.hw_threshold)]
            else:
                x = x[x[:, 4] > self.conf_threshold]  # confidence
            # If none remain process next image
            if len(x) == 0:
                continue
            # Compute conf
            x[:, 5:5+self.n_classes] *= x[:, 4][:, None]  # conf = obj_conf * cls_conf

            max_conf, max_i = x[:, 5:5+self.n_classes].max(1, keepdim=True)
            if self.is_ro:
                _, degree = x[:, 5+self.n_classes:].max(1, keepdim=True)
                degree = degree.float() * 90 / self.n_degree
                x = torch.cat((x[:, :4], degree, max_conf, max_i.float()), 1)[max_conf.view(-1) > self.conf_threshold]
            else:
                x = torch.cat((x[:, :4], max_conf, max_i.float()), 1)[max_conf.view(-1) > self.conf_threshold]
            # Check shape
            n = len(x)  # number of boxes
            if n == 0:  # no boxes
                continue
            elif n > self.max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:self.max_nms]]  # sort by confidence
            # nms
            if self.is_ro:
                nms_res = self._nms_ro(x[:, :5], x[:, 5], x[:, 6])
            else:
                box = self.xywh2xyxy(x[:, :4])
                if self.batched_nms:
                    nms_res = torchvision.ops.batched_nms(box, x[:, 4], x[:, 5], self.iou_threshold)
                else:
                    nms_res = torchvision.ops.nms(box, x[:, 4], self.iou_threshold)
            if nms_res.shape[0] > self.max_det:  # limit detections
                nms_res = nms_res[:self.max_det]
            # something else
            res[batch_i] = x[nms_res].cpu().numpy()
            if (time.time() - t) > self.patience:
                print(f'WARNING: NMS time limit {self.patience}s exceeded')
                break  # time limit exceeded
        if self.extra_filter:
            res = self.extra_filter(res)
        return res

    def _nms_ro(self, boxes, conf, class_id):
        '''
        Calculate NMS for robndbox
        boxex: rotated bounding boxes; torch.Tensor or ndarray; shape: [num of robndbox, 5];
               5 features: cx, cy, cw, ch, degree; ratio values.
        conf: confidences; torch.Tensor; shape: [num of robndbox,]
        class_id: class id; torch.Tensor; shape: [num of robndbox,]
        RETURN: indices list of kept rotated bounding boxes.
                mask matrix if keep_mask is True.
        '''
        # pdb.set_trace()
        n = len(boxes)
        if type(boxes) == np.ndarray:
            masks = np.zeros((n, self.mask_h, self.mask_w), dtype='uint8')
            is_torch = False
        else:
            masks = torch.zeros((n, self.mask_h, self.mask_w), dtype=torch.int8, device=boxes.device)
            is_torch = True
        # ratio values
        boxes = self.xywh2xyxy(boxes)
        kept = [True] * n
        mask_checked = [False] * n
        for i in range(n-1):
            if not kept[i]:
                continue
            if self.extra_filter:
                if class_id[i] in [0, 2, 26]:
                    if min(boxes[i][2] - boxes[i][0],
                           boxes[i][3] - boxes[i][1]) > 1.2 * self.extra_filter.record[f'{int(class_id[i])}_short'] or \
                       min(boxes[i][2] - boxes[i][0],
                           boxes[i][3] - boxes[i][1]) < 0.5 * self.extra_filter.record[f'{int(class_id[i])}_short']:
                        kept[i] = False
                        continue
                elif class_id[i] in [10]:
                    if max(boxes[i][2] - boxes[i][0],
                           boxes[i][3] - boxes[i][1]) < 0.8*self.extra_filter.record[f'{int(class_id[i])}_long']:
                        kept[i] = False
                        continue
                elif class_id[i] in [18]:
                    if max(boxes[i][2] - boxes[i][0],
                           boxes[i][3] - boxes[i][1]) < 0.8*self.extra_filter.record[f'{int(class_id[i])}_long']:
                        kept[i] = False
                        continue
            if not mask_checked[i]:
                self._get_mask(masks, i, boxes[i], is_torch)
                mask_checked[i] = True
            for j in range(i+1, n):
                if not kept[j]:
                    continue
                if not mask_checked[j]:
                    self._get_mask(masks, j, boxes[j], is_torch)
                    mask_checked[j] = True
                if self.calc_iou_ro(masks[i], masks[j], is_torch) > self.iou_threshold:
                    if conf[i] >= conf[j]:
                        kept[j] = False
                        continue
                    else:
                        kept[i] = False
                        break
        return np.arange(n)[kept]

    def _get_mask(self, masks, index, box, is_torch):
        '''
        Get a 0-1 mask for box; inplace maneuver.
        masks: torch.Tensor or ndarray; shape: [num of robndbox, mask_h, mask_w]; inplace;
        index: mask index; temp param;
        box: robndboxes; torch.Tensor or ndarray; shape: [5,];
               5 features: xmin, ymin, xmax, ymax, degree; ratio values.
        is_torch: if True, pytorch version, otherwise, numpy version.
        ATTENTION: we define this consensus
        0----1
        |    |
        3----2
        '''
        # pdb.set_trace()
        if is_torch:
            device = masks.device
            tan = lambda temp_degree: torch.tan(temp_degree * pi/180)
            box = (box * torch.tensor([self.mask_w, self.mask_h, self.mask_w, self.mask_h, 1],
                                      device=device)).long()
            rotate_p_list = rotate_p_list_pt
        else: # ndarray
            tan = lambda temp_degree: np.tan(temp_degree * pi/180)
            # ATTENTION: take care of the overflow; uint16 ranges from 0 ~ 65535 which is sure enough for the mask.
            # box = (box * [self.mask_w, self.mask_h, self.mask_w, self.mask_h, 1]).astype('uint16')
            box = (box * [self.mask_w, self.mask_h, self.mask_w, self.mask_h, 1]).astype(int)
            rotate_p_list = rotate_p_list_np
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        degree = box[-1]
        if degree == 0:
            masks[index,
                  box[1]: box[3]+1,
                  box[0]: box[2]+1] = 1
        else:
            p_list = [[box[0], box[1]],
                      [box[2], box[3]]]
            p_list_ro = rotate_p_list(p_list, cx, cy, degree)
            a0 = tan(degree)
            b_tr = p_list_ro[0][1] - a0 * p_list_ro[0][0]
            b_bl = p_list_ro[1][1] - a0 * p_list_ro[1][0]
            a1 = tan(degree+90)
            b_tl = p_list_ro[0][1] - a1 * p_list_ro[0][0]
            b_br = p_list_ro[1][1] - a1 * p_list_ro[1][0]
            if is_torch:
                x = torch.arange(self.mask_w, device=device)[None, ...]
                y = torch.arange(self.mask_h, device=device)[..., None]
            else:
                y, x = np.ogrid[:self.mask_h, :self.mask_w]
            masks[index,
                  (y > (a0 * x + b_tr)) & (y < (a0 * x + b_bl))
                  & (y > (a1 * x + b_tl)) & (y < (a1 * x + b_br))] = 1

    def get_mask(self, box):
        '''
        Get single mask from box.
        Given box: robndboxes; tensor or ndarray; shape: [5,];
                   5 features: xmin, ymin, xmax, ymax, degree; ratio values.
        Return the corresponding mask map in shape of [mask_h, mask_w].
        '''
        if type(box) == np.ndarray:
            masks = np.zeros((1, self.mask_h, self.mask_w), dtype='uint8')
            is_torch = False
        else:
            masks = torch.zeros((1, self.mask_h, self.mask_w), dtype=torch.int8, device=box.device)
            is_torch = True
        self._get_mask(masks, 0, box, is_torch)
        return masks[0]

    @staticmethod
    def calc_iou_ro(mask0, mask1, is_torch, eps=1e-7):
        '''
        numpy version
        Itersection over Union for rotated bounding boxes.
        mask0, mask1: 0-1 mask of robndbox; shape: (mask_h, mask_w)
        is_torch: if True, pytorch version, otherwise, numpy version.
        RETURN: iou value
        ************** TO BE CONTINUED: use min max range to get smaller mask area ***************
        '''
        if is_torch:
            temp_sum = torch.sum
        else:
            temp_sum = np.sum
        inter = temp_sum(mask0 * mask1)
        union = temp_sum(mask0) + temp_sum(mask1) - inter
        return inter / (union + eps)

    @staticmethod
    def xywh2xyxy(pred):
        '''
        ndarrya or torch.Tensor
        pred: pred[i] output from post_process()
              shape = [num of predicted boxes, 5 + nc + n_degree]
                       5 features: cx, cy, cw, ch, conf
                       cx, cy, cw, ch are all percentage values with respect to the abs width and height.
        RETURN: a same shaped matrix but with cx, cy, cw, ch replaced by xmin, ymin, xmax, ymax,
                xmin, ymin, xmax, ymax are all percentage values with respect to the abs width and height.
        '''
        res = deepcopy(pred)
        if len(res.shape) == 1:
            res[0] = pred[0] - pred[2] / 2
            res[1] = pred[1] - pred[3] / 2
            res[2] = pred[0] + pred[2] / 2
            res[3] = pred[1] + pred[3] / 2
        else:
            res[:, 0] = pred[:, 0] - pred[:, 2] / 2
            res[:, 1] = pred[:, 1] - pred[:, 3] / 2
            res[:, 2] = pred[:, 0] + pred[:, 2] / 2
            res[:, 3] = pred[:, 1] + pred[:, 3] / 2
        return res

    @staticmethod
    def calc_iou(box0, box1, eps=1e-7):
        '''
        numpy version
        Itersection over Union calculation
        box0, box1: shape could be [4,] or [num_target, 4]
                    cx,cy,cw,ch could be abs values or percentage values.
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
        inter = np.max([np.min([b0_xmax, b1_xmax], axis=0) - np.max([b0_xmin, b1_xmin], axis=0),
                        np.zeros_like(b0_xmax)], axis=0) * \
                np.max([np.min([b0_ymax, b1_ymax], axis=0) - np.max([b0_ymin, b1_ymin], axis=0),
                        np.zeros_like(b0_ymax)], axis=0)
        # Union Area
        union = b0_cw * b0_ch + b1_cw * b1_ch - inter

        return inter / (union + eps)


class Metric:
    '''
    ATTENTION:
    This is a numpy version.
    Each class id has a metric matrix for different iou_thresholds.
    We calculate precision, recall, f1 score, true_positive,
    pred_true(num of predicted boxes with confidence >= conf_threshold) for each conf_threshold.
    n_target_box(num of bbox) doesn't change with the conf_threshold and iou_thresholds.
        It is duplicated just for alignment.
    ap(average precision) is calculated for each class id on each iou_threshold,
        only target classes are taken into account.
    n_cls+1: the last one is to save metrics on the whole bunch of boxes.
    '''

    def __init__(self, n_cls, is_ro, cfg, n_degree=90):
        '''
        n_cls: number of classes
        is_ro: if True, for rotated bounding boxes.
        cfg: dict; nms config.
        n_degree: option; number of degrees for rotated bounding box.
        '''
        self.n_cls = n_cls
        self.is_ro = is_ro
        self.iou_thresholds = np.arange(start=0.5, stop=0.96, step=0.05).round(2)
        self.conf_thresholds = np.arange(start=0.25, stop=0.96, step=0.05).round(2)
        self.n_iou = len(self.iou_thresholds)
        self.n_conf = len(self.conf_thresholds)
        for name in ['precision', 'recall', 'f1', 'true_positive', 'pred_true']:
            setattr(self, name, np.zeros((n_cls + 1, self.n_iou, self.n_conf)))
        self.n_target_box = np.zeros((n_cls + 1))
        self.seen_target_cls = set()
        self.ap = None
        self.mAPs = None
        self.mAP = None
        self.nms = NMS(n_classes=n_cls,
                       is_ro=is_ro,
                       n_degree=n_degree,
                       **cfg)

    def update(self, pred, target):
        '''
        pred (torch.Tensor):
            output from post_process()
            shape = [batch size, num of predicted boxes, 5+n_classes+n_degree]
                5 features: cx, cy, cw, ch, conf.
                cx, cy, cw, ch are all percentage values with respect to the abs width and height.
        target (torch.Tensor) :
            shape = [num of bboxes, 6 or 7]
                7 features: batch index, cls id, cx, cy, cw, ch, degree
                cx, cy, cw, ch are all percentage values with respect to the abs width and height.
        '''
        pred = self.nms(pred)
        target = target.cpu().numpy()
        n_target = len(target)
        t_checked = [False] * n_target

        if self.is_ro:
            target_masks = np.zeros((n_target, self.nms.mask_h, self.nms.mask_w), dtype='uint8')
            target_boxes = self.nms.xywh2xyxy(target[:, -5:])
            target_masks_checked = [False] * n_target

        n_target_boxes = Counter(target[:, 1].astype(int))
        for key, value in n_target_boxes.items():  # update the global self.n_target_box
            self.n_target_box[key] += value
            self.n_target_box[-1] += value
            self.seen_target_cls.add(key)

        for i_batch in range(len(pred)):
            if isinstance(pred[i_batch], int):
                continue
            for i_pred, pred_i in enumerate(pred[i_batch]):
                cls_id = int(pred_i[-1])
                where_conf = pred_i[-2] >= self.conf_thresholds
                if not np.any(where_conf):
                    continue
                self.pred_true[cls_id, :, where_conf] += 1
                self.pred_true[-1, :, where_conf] += 1

                if self.is_ro:
                    pred_box = self.nms.xywh2xyxy(pred_i[:5])
                    pred_mask = self.nms.get_mask(pred_box)

                for i_target in np.where(target[:, 1] == cls_id)[0]:
                    if t_checked[i_target]:
                        continue
                    if self.is_ro:
                        if not target_masks_checked[i_target]:
                            self.nms._get_mask(target_masks,
                                               i_target,
                                               target_boxes[i_target],
                                               is_torch=False)
                            target_masks_checked[i_target] = True
                        iou = self.nms.calc_iou_ro(target_masks[i_target], pred_mask, is_torch=False)
                    else:
                        iou = self.nms.calc_iou(pred_i[:4], target[i_target][-4:])
                    where_iou = iou >= self.iou_thresholds
                    if np.any(where_iou):
                        t_checked[i_target] = True
                        # I don't get it. How to fix this ?
                        temp = self.true_positive[cls_id, where_iou]
                        temp[:, where_conf] += 1
                        self.true_positive[cls_id, where_iou] = temp
                        temp = self.true_positive[-1, where_iou]
                        temp[:, where_conf] += 1
                        self.true_positive[-1, where_iou] = temp
                        # one v.s. one mode:
                        # One predicted box only for one target but not one for multi-targets
                        break
        return

    def calc_metrics(self):
        '''
        Do the metrics calculations.
        The average precision (AP) is a way to summarize the
        precision-recall curve into a single value representing the average of
        all precisions on different confidence threshold.
        The AP is the weighted sum of precisions at each threshold where the weight is the increase in recall.
        The mAPs for each iou threshold is the mean value of APs for all classes.
        The mAP is the mean value of all mAPs on different iou thresholds.
        VOC only cares about mAP(iou_threshold = 0.5).
        COCO takes into account of mAPs(iou_threshold = 0.5, 0.55, ..., 0.95) and mAP = mean(mAPs).
        ATTENTION:
        self.ap[-1] is for the whole bunch of classes.
        '''
        self.calc_precision()
        self.calc_recall()
        self.calc_f1()
        recalls = np.append(self.recall, np.zeros((self.n_cls + 1, self.n_iou, 1)), axis=-1)
        self.ap = np.sum((recalls[..., :-1] - recalls[..., 1:]) * self.precision, axis=-1)
        # mAP only takes target cls_ids into account
        self.mAPs = {self.iou_thresholds[i]: value for i, value in \
                     enumerate(np.mean(self.ap[list(self.seen_target_cls)] if self.seen_target_cls else self.ap,
                                       axis=0))}
        self.mAP = np.mean(list(self.mAPs.values()))
        # Another unofficial way to calculate mAP is like this:
        # self.mAP = np.mean(self.ap[-1])


    def calc_precision(self):
        '''
        Precision calculation
        '''
        self.precision[np.logical_and(self.true_positive == 0, self.pred_true == 0)] = 1
        self.precision[np.logical_and(self.true_positive != 0, self.pred_true == 0)] = 0
        other_places = self.pred_true != 0
        self.precision[other_places] = self.true_positive[other_places] / self.pred_true[other_places]

    def calc_recall(self):
        '''
        Recall calculation
        '''
        n_target_box = np.repeat(self.n_target_box, self.n_iou * self.n_conf) \
            .reshape(self.n_cls + 1, self.n_iou, self.n_conf)
        self.recall[np.logical_and(self.true_positive == 0, n_target_box == 0)] = 1
        self.recall[np.logical_and(self.true_positive != 0, n_target_box == 0)] = 0
        other_places = n_target_box != 0
        self.recall[other_places] = self.true_positive[other_places] / n_target_box[other_places]

    def calc_f1(self):
        '''
        F1 score calculation
        '''
        self.f1[np.logical_or(self.precision == 0, self.recall == 0)] = 0
        other_places = np.logical_and(self.precision != 0, self.recall != 0)
        self.f1[other_places] = 2 * self.precision[other_places] * self.recall[other_places] / \
                                (self.precision[other_places] + self.recall[other_places])


class ExtraFilter:
    '''
    Filter with the real world size parameters.
    '''
    def __init__(self, record_path, threshold=0.80):
        '''
        record_path: extra filter size; json file
        threshold: conf threshold less than which we will measure the size
        '''
        self.record = json_read(record_path)
        self.threshold = threshold

    def __call__(self, pred):
        '''
        pred: (list of ndarray) len == batch size; shape = [num of predicted boxes after nms, 6 or 7]
                6 or 7 features: cx, cy, cw, ch, (degree), confidence, cls_id
                cx, cy, cw, ch are all ratio values with respect to the abs width and height.
        '''
        # pdb.set_trace()
        for batch_i in range(len(pred)):
            if isinstance(pred[batch_i], int):
                continue
            suspects = np.where((pred[batch_i][:, -1] == 12) |
                                (pred[batch_i][:, -1] == 23) |
                                (pred[batch_i][:, -1] == 16) |
                                (pred[batch_i][:, -1] == 17))[0].tolist()
            for box_i in suspects:
                if pred[batch_i][box_i][-2] < self.threshold:
                    x_factor = np.sqrt(pred[batch_i][box_i][2] ** 2 + pred[batch_i][box_i][3] ** 2)
                    # x_factor = pred[batch_i][box_i][2] * pred[batch_i][box_i][3]
                    if int(pred[batch_i][box_i][-1]) in [12, 23]:
                        if abs(x_factor - self.record['12']) < abs(x_factor - self.record['23']):
                            pred[batch_i][box_i][-1] = 12.0
                        else:
                            pred[batch_i][box_i][-1] = 23.0
                    elif int(pred[batch_i][box_i][-1]) in [16, 17]:
                        if abs(x_factor - self.record['16']) < abs(x_factor - self.record['17']):
                            pred[batch_i][box_i][-1] = 16.0
                        else:
                            pred[batch_i][box_i][-1] = 17.0
        return pred