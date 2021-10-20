import pdb
import cv2
import numpy as np
import random
from tools.utils import (label_xywh2xyxy, mkdirs, label_on_table,
                         pad_rotate, get_robndbox_from_points, ro_label_on_table)
from tools.const import *
import os
import glob
from tools.compress import compress
from PIL import Image
import h5py

class Augment:
    '''
    Augmentation class.
    If an item in aug_list is a list, for each time,
    we will randomly choose one operator in this list to play with.
    '''
    def __init__(self, aug_list):
        '''
        aug_list: augmentation operators. 
        '''
        self.aug_list = aug_list
        return

    def __call__(self, img, label):
        '''
        img: (height, width, channel) RGB
        label: targets in yolo format: (num of objs, 5) class, cx, cy, cw, ch
        '''
        for aug in self.aug_list:
            if isinstance(aug, list):
                aug = random.choice(aug)
            img, label = aug(img, label)
        return img, label


class Resize:
    '''
    As the name indicates.
    ATTENTION: This only works for bndbox, not for robndbox.
    '''
    def __init__(self, h=None, w=None):
        '''
        h, w: target image shape
        '''
        self.h = h
        self.w = w

    def __call__(self, img, label, target_h=None, target_w=None):
        h, w = img.shape[:2]
        self.h = target_h or self.h
        self.w = target_w or self.w
        if self.h / h == 1 and self.w / w == 1:
            return img, label
        elif self.h / h < 1 or self.w / w < 1:
            return cv2.resize(img, (self.w, self.h), cv2.INTER_AREA), label
        else:
            return cv2.resize(img, (self.w, self.h), cv2.INTER_LINEAR), label


class PadResize:
    '''
    Resize and pad pad with 0 or background img.
    ATTENTION: It works for both bndbox and robndbox.
    '''
    def __init__(self, background=None, h=None, w=None, sym=False, frozen_shrink=False, return_pad=False):
        '''
        h, w: target height and weight, just in case of being invoked by Augment() in val and inference.
        background: instance of Class Background.
        sym: symmetrical padding or not.
        frozen_shrink: if True, shrink the image and pad to the original shape, specifically for Shrink().
        return_pad: if True, if background is None, return [top, bottom, left, right] pad size.
        '''
        self.h = h
        self.w = w
        self.background = background
        self.sym = sym
        self.frozen_shrink = frozen_shrink
        self.return_pad = return_pad

    def __call__(self, img, label=[], h=None, w=None, return_pad=False):
        '''
        img: (height, width, channel)
        label: targets in yolo format: (num of objs, 5 or 6) class, cx, cy, cw, ch (,degree); ratio values.
               It works for both bndbox and robndbox.
        h, w: resized shape; priority is higher than self.h, self.w

        '''
        # pdb.set_trace()
        h = h or self.h
        w = w or self.w
        img_h, img_w = img.shape[:2]
        ratio = min(h / img_h, w / img_w)
        resized_h = int(img_h * ratio)
        resized_w = int(img_w * ratio)
        if ratio < 1:
            img = cv2.resize(img, (resized_w, resized_h), cv2.INTER_AREA)
        elif ratio > 1:
            img = cv2.resize(img, (resized_w, resized_h), cv2.INTER_LINEAR)

        if self.frozen_shrink:
            h = img_h
            w = img_w

        left = (w - resized_w)//2 if self.sym else int(random.random() * (w - resized_w))
        top = (h - resized_h)//2 if self.sym else int(random.random() * (h - resized_h))
        for l in label:
            l[1] = (l[1] * resized_w + left) / w
            l[2] = (l[2] * resized_h + top) / h
            l[3] = l[3] * resized_w / w
            l[4] = l[4] * resized_h / h
        if self.background:
            back_img = self.background.get_background((h, w, 3))
            back_img[top:top + resized_h, left:left + resized_w, :] = img
            return back_img, label
        else:
            right = w - resized_w - left
            bottom = h - resized_h - top
            img = cv2.copyMakeBorder(img, top, bottom, left, right, 0, 0)
            if self.return_pad:
                return img, label, [top, bottom, left, right]
            else:
                return img, label


class Shrink:
    '''
    Prerequisite: The original image has already been resized into the right size.
    1. Randomly shrink image.
    2. Pad to the original shape with 0 or background img.
    ATTENTION: img shape doesn't change anyway.
               It works for both bndbox and robndbox.
    '''
    def __init__(self, background=None, lower_bound=0.75, sym=False, rand_jump=True):
        '''
        background: instance of Background, if None, make border with 0.
        lower_bound: lower bound of shrink ratio, ex: shrink ratio ranges from 0.65 to 1.
        sym: symmetrical padding or not.
        rand_jump: if True, randomly jump over the Mosaic augmentation.
        '''
        self.background = background
        self.lower_bound = lower_bound
        self.rand_jump = rand_jump
        self.pad_resize = PadResize(background=background, sym=sym, frozen_shrink=True)

    def __call__(self, img, label):
        '''
        img: (height, width, channel)
        label: targets in yolo format: (num of objs, 5) class, cx, cy, cw, ch; ratio values.
               It works for both bndbox and robndbox.
        '''
        if self.rand_jump and random.random() < 0.5:
            return img, label
        h, w = img.shape[:2]
        ratio = 1 - random.random() * (1 - self.lower_bound)
        return self.pad_resize(img, label, int(h*ratio), int(w*ratio))


class PadRotate:
    '''
    Rotate and pad with 0 to keep the whole rotated image. Then PadResize to the original size.
    ATTENTION: img shape doesn't change anyway.
               It only works for robndbox, not for bndbox.
    '''
    def __init__(self, background=None, rand_jump=True, sym=False):
        '''
        background: instance of Class Background.
        rand_jump: if True, randomly jump over the Mosaic augmentation.
        sym: symmetrical padding or not.
        '''
        self.rand_jump = rand_jump
        self.pad_resize = PadResize(background=background, sym=sym)

    def __call__(self, img, label, degree=None):
        '''
        img: (height, width, channel)
        label: ndarray; targets in yolo format: (num of objs, 6) class, cx, cy, cw, ch, degree; ratio values.
               It only works for robndbox, not for bndbox.
        '''
        if self.rand_jump and random.random() < 0.5:
            return img, label
        degree = degree or random.randint(0, 360)
        h, w = img.shape[:2]
        img, trans_matrix = pad_rotate(img, degree)
        new_h, new_w = img.shape[:2]
        del_list = []
        for i in range(len(label)):
            rect = label[i][1:]
            rect = ((rect[0] * w, rect[1] * h), (rect[2] * w, rect[3] * h), rect[4])
            corners = cv2.boxPoints(rect)
            new_corners = trans_matrix.dot(np.row_stack([corners.T, [1, 1, 1, 1]])).T
            rect = get_robndbox_from_points(new_corners)
            if rect:
                label[i][1] = rect[0][0] / new_w
                label[i][2] = rect[0][1] / new_h
                label[i][3] = rect[1][0] / new_w
                label[i][4] = rect[1][1] / new_h
                label[i][5] = rect[2]
            else:
                del_list.append(i)
        if del_list:
            label = np.delete(label, del_list, 0)
        return self.pad_resize(img, label, h=h, w=w)


class Mov:
    '''
    Randomly move the img and pad with 0 or background img.
    ATTENTION: img shape doesn't change anyway.
               It works for both bndbox and robndbox.
    '''
    def __init__(self, background=None, upper_bound=0.25,
                 threshold=0.04, is_ro=True, rand_jump=True):
        '''
        background: instance of Background, if None, make border with 0.
        upper_bound: largest movement ratio.
        threshold: robndbox edge length smaller than threshold would be ignored.
        is_ro: if True, for rotated bounding boxes.
        rand_jump: if True, randomly jump over the Mosaic augmentation.
        '''
        self.background = background
        self.upper_bound = upper_bound
        self.threshold = threshold
        self.is_ro = is_ro
        self.rand_jump = rand_jump

    def __call__(self, img, label, mov_x=None, mov_y=None):
        '''
        img: (height, width, channel)
        label: targets in yolo format: (num of objs, 6) class, cx, cy, cw, ch, degree
        mov_x, mov_y: move distance
        '''
        if self.rand_jump and random.random() < 0.5:
            return img, label
        # pdb.set_trace()
        shape = img.shape
        h, w = shape[:2]

        mov_x = mov_x or int(random.random() * self.upper_bound * w * random.choice([-1, 1]))
        mov_y = mov_y or int(random.random() * self.upper_bound * h * random.choice([-1, 1]))

        if self.is_ro:
            # for robndbox
            if len(label) > 0:
                label[:, 1] = label[:, 1] * w + mov_x
                label[:, 2] = label[:, 2] * h + mov_y
                label[:, 3] *= w
                label[:, 4] *= h
                label = ro_label_on_table(label, [0, 0, w-1, h-1], self.threshold)
        else:
            # for bndbox
            label_xywh2xyxy(label, h, w)
            del_list = []
            table = [0, 0, w - 1, h - 1]
            for i, l in enumerate(label):
                l = label_on_table([l[0],
                                    l[1] + mov_x,
                                    l[2] + mov_y,
                                    l[3] + mov_x,
                                    l[4] + mov_y],
                                   table)
                if l is None:
                    del_list.append(i)
                else:
                    label[i] = l
            label = np.delete(label, del_list, 0)

        img = img[max(0, -mov_y): min(h, h-mov_y),
                  max(0, -mov_x): min(w, w-mov_x), :]

        if self.background:
            back_img = self.background.get_background(shape)
            back_img[max(0, mov_y): min(h, h+mov_y),
                     max(0, mov_x): min(w, w+mov_x), :] = img
            return back_img, label
        else:
            if mov_x >= 0:
                left = mov_x
                right = 0
            else:
                left = 0
                right = mov_x
            if mov_y >= 0:
                top = mov_y
                bottom = 0
            else:
                top = 0
                bottom = mov_y
            img = cv2.copyMakeBorder(img, top, bottom, left, right, 0, 0)
            return img, label


class Crop:
    '''
    Do the cropping augmentation:
    1. Randomly choose one part from the image.
    2. Then enlarge the img back into the original size.
    Resize to the same size as the input image.
    ATTENTION: img shape doesn't change anyway.
               It works for both bndbox and robndbox.
    '''

    def __init__(self, background=None, crop_w_min_p=0.75, crop_h_min_p=0.75,
                 sym=False, threshold=0.04, is_ro=True, rand_jump=True):
        '''
        background: instance of Background, if None, make border with 0.
        crop_w_min_p: cw of the cropped area ranges from crop_w_min_p to 1
        crop_h_min_p: ch of the cropped area ranges from crop_w_min_p to 1
        sym: symmetrical padding or not.
        threshold: robndbox edge length smaller than threshold would be ignored.
        is_ro: if True, for rotated bounding boxes.
        rand_jump: if True, randomly jump over the Mosaic augmentation.
        '''
        self.crop_w_min_p = crop_w_min_p
        self.crop_h_min_p = crop_h_min_p
        self.threshold = threshold
        self.is_ro = is_ro
        self.rand_jump = True
        self.pad_resize = PadResize(background=background, sym=sym)

    def __call__(self, img, label):
        '''
        img: (height, width, channel)
        label: targets in yolo format: (num of objs, 5) class, cx, cy, cw, ch
        '''
        if self.rand_jump and random.random() < 0.5:
            return img, label
        h, w = img.shape[:2]
        crop_w_p = random.random() * (1 - self.crop_w_min_p) + self.crop_w_min_p
        crop_h_p = random.random() * (1 - self.crop_h_min_p) + self.crop_h_min_p
        crop_w = int(crop_w_p * w)
        crop_h = int(crop_h_p * h)
        xmin = int(random.random() * (1 - crop_w_p) * w)
        ymin = int(random.random() * (1 - crop_h_p) * h)
        xmax = xmin + crop_w -1
        ymax = ymin + crop_h -1

        img = img[ymin: ymax + 1, xmin: xmax + 1, :]

        if self.is_ro:
            if len(label) > 0:
                # for robndbox
                label[:, 1] *= w
                label[:, 2] *= h
                label[:, 3] *= w
                label[:, 4] *= h
                label = ro_label_on_table(label, [xmin, ymin, xmax, ymax], self.threshold)
            return self.pad_resize(img, label, h, w)
        else:
            # for bndbox
            label_xywh2xyxy(label, h, w)
            del_list = []
            table = [xmin, ymin, xmax, ymax]
            for i, l in enumerate(label):
                l = label_on_table(l, table)
                if l is None:
                    del_list.append(i)
                else:
                    label[i] = l
            label = np.delete(label, del_list, 0)
            img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
            return img, label


class Flip:
    '''
    Flip up side down or left side right.
    ATTENTION: It works for both bndbox and robndbox.
    '''
    def __init__(self, t_ud=0.5, t_lr=0.5, is_ro=True, rand_jump=True):
        '''
        t_ud: threshold for up side down
        t_lr: threshold for left side right
        is_ro: if True, for rotated bounding boxes.
        rand_jump: if True, randomly jump over the Mosaic augmentation.
        '''
        self.t_ud = t_ud
        self.t_lr = t_lr
        self.is_ro = is_ro
        self.rand_jump = rand_jump

    def __call__(self, img, label):
        '''
        img: (height, width, channel)
        label: targets in yolo format: (num of objs, 5) class, cx, cy, cw, ch
        '''
        if self.rand_jump and random.random() < 0.5:
            return img, label
        h, w = img.shape[:2]
        flag_ud = False
        flag_lr = False
        if random.random() < self.t_ud:
            img = np.flipud(img)
            if len(label) != 0:
                if self.is_ro:
                    flag_ud = True
                else:
                    label[:, 2] = 1 - label[:, 2]
        if random.random() < self.t_lr:
            img = np.fliplr(img)
            if len(label) != 0:
                if self.is_ro:
                    flag_lr = True
                else:
                    label[:, 1] = 1 - label[:, 1]
        if flag_lr or flag_ud:
            del_list = []
            for i in range(len(label)):
                rect = label[i][1:]
                rect = ((rect[0]*w, rect[1]*h), (rect[2]*w, rect[3]*h), rect[4])
                corners = cv2.boxPoints(rect)
                if flag_ud:
                    corners[:, 1] = h - 1 - corners[:, 1]
                if flag_lr:
                    corners[:, 0] = w - 1 - corners[:, 0]
                rect = get_robndbox_from_points(corners)
                if rect:
                    label[i][1] = rect[0][0] / w
                    label[i][2] = rect[0][1] / h
                    label[i][3] = rect[1][0] / w
                    label[i][4] = rect[1][1] / h
                    label[i][5] = rect[2]
                else:
                    del_list.append(i)
            if del_list:
                label = np.delete(label, del_list, 0)
        return img, label


class HsvAug:
    '''
    Augmentation on hue, saturation and value.
    ***** Borrowed from official yolov5 *****
    ATTENTION: It works for both bndbox and robndbox.
    '''
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, rand_jump=True):
        '''
        hgain: gain on hue
        sgain: gain on saturation
        vgain: gain on value
        rand_jump: if True, randomly jump over the Mosaic augmentation.
        '''
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.rand_jump = rand_jump

    def __call__(self, img, label):
        '''
        img: (height, width, channel) RGB
        label: targets in yolo format: (num of objs, 5) class, cx, cy, cw, ch
        * label doesn't change anyway.
        '''
        if self.rand_jump and random.random() < 0.5:
            return img, label
        img = img.astype(np.uint8)
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # 0.5~1.5
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)  # no return needed
        return img, label


class Mosaic:
    '''
    Mosaic augmentation.
    ATTENTION: It works for both bndbox and robndbox.
    '''
    def __init__(self, h5_path, id_list, background=None, ratio=0.1, h=640, w=640, rand_jump=True,
                 is_ro=True, threshold=0.04):
        '''
        h5_path: dataset.h5 file path
        id_list: filenames in .h5
        background: instance of Background, if None, make border with 0.
        ratio: maximum movement proportion of center point.
        h: target height
        w: target width
        rand_jump: if True, randomly jump over the Mosaic augmentation.
        is_ro: if True, for rotated bounding boxes.
        threshold: robndbox edge length smaller than threshold would be ignored.
        '''
        self.h5_path = h5_path
        self.id_list = id_list
        self.ratio = ratio
        self.h = h
        self.w = w
        self.background = background
        self.rand_jump = rand_jump
        self.is_ro = is_ro
        self.threshold = threshold
        self.resize = PadResize(background=background, h=self.h, w=self.w) if is_ro else Resize(h=self.h, w=self.w)
        self.pad_resize = PadResize(background=background, h=self.h, w=self.w)

    def size_fit(self, img, label):
        '''
        For img larger than required.
        Shrink the image to fit the size of self.h * self.w
        '''
        h, w = img.shape[:2]
        ratio = min(self.h/h, self.w/w)
        if ratio < 1:
            img, label = self.resize(img, label)
        return img, label

    def _one_quarter(self, back_img, img=None, label=None, up=True, left=True):
        '''
        Paradigm of filling one quarter of the background with one image.
        back_img: background image.
        img: image to fill the background, [height, width, channel] RGB
        label: yolo format [class, cx,cy,cw,ch].
        up, left: decides which part is this quarter at.
        * up=False is down, left=False is right.
        '''
        if img is None:
            with h5py.File(self.h5_path, 'r') as f:
                i = random.choice(self.id_list)
                img = np.asarray(Image.open(f[i].attrs['img']))
                label = np.asarray(f[i]['label'])
                img, label = self.size_fit(img, label)
                # if len(label) == 0:
                #     label = np.repeat(label[:, None], 5, axis=-1)
        h, w = img.shape[:2]
        dx = self.cx - (w if left else 0)
        dy = self.cy - (h if up else 0)
        labels = []
        if self.is_ro:
            if len(label) > 0:
                # for robndbox
                label[:, 1] = label[:, 1] * w + dx
                label[:, 2] = label[:, 2] * h + dy
                label[:, 3] *= w
                label[:, 4] *= h
                label = ro_label_on_table(label, [0, 0, 2*self.w-1, 2*self.h-1], self.threshold)
            labels += label.tolist()
        else:
            # for bndbox
            label_xywh2xyxy(label, h, w)
            for l in label:
                l = label_on_table([l[0],
                                    l[1] + dx,
                                    l[2] + dy,
                                    l[3] + dx,
                                    l[4] + dy],
                                   [0, 0, 2*self.w-1, 2*self.h-1])
                if l is not None:
                    labels.append(l)
        back_range_h = (max(0, self.cy - h), self.cy) if up else (self.cy, min(2*self.h, self.cy + h))
        back_range_w = (max(0, self.cx - w), self.cx) if left else (self.cx, min(2*self.w, self.cx + w))
        img_range_h = (max(0, h - self.cy), h) if up else (0, min(2*self.h - self.cy, h))
        img_range_w = (max(0, w - self.cx), w) if left else (0, min(2*self.w - self.cx, w))
        back_img[back_range_h[0]:back_range_h[1],
                 back_range_w[0]:back_range_w[1], :] = img[img_range_h[0]:img_range_h[1],
                                                           img_range_w[0]:img_range_w[1], :]
        return labels

    def __call__(self, img, label):
        '''
        img: (height, width, channel)
        label: targets in yolo format: (num of objs, 5) class, cx, cy, cw, ch
        '''

        if self.rand_jump:
            luck = random.choice(range(4))
            if luck == 1:
                return self.resize(img, label)
            elif luck == 2:
                return self.pad_resize(img, label)
        img, label = self.size_fit(img, label)
        shape = (2*self.h, 2*self.w, 3)
        if self.background:
            back_img = self.background.get_background(shape)
        else:
            back_img = np.zeros(shape)
#         pdb.set_trace()
        self.cx = int((0.5 + random.random() * self.ratio * random.choice([1, -1])) * 2*self.w)
        self.cy = int((0.5 + random.random() * self.ratio * random.choice([1, -1])) * 2*self.h)
        labels = []
        labels.extend(self._one_quarter(back_img, img, label, up=True, left=True))
        labels.extend(self._one_quarter(back_img, up=True, left=False))
        labels.extend(self._one_quarter(back_img, up=False, left=True))
        labels.extend(self._one_quarter(back_img, up=False, left=False))
        return self.resize(back_img, np.asarray(labels))


class Background:
    '''
    Background generation.
    '''
    def __init__(self, background_source=BACKGROUND_SOURCE,
                 background_folder=BACKGROUND_FOLDER,
                 zero_threshold=0.75,
                 overwrite=False):
        '''
        background_source: source background images.
        background_folder: compressed background images.
        zero_threshold: threshold over which we set the back pure black.
        overwrite: if True, redo the generating process.
        '''
        if not os.path.exists(background_folder) or overwrite:
            mkdirs(background_folder)
            for i in tqdm(glob.glob(os.path.join(background_source, '*')),
                         desc=f'Compressing background img into {background_folder}'):
                compress(i, os.path.join(background_folder, os.path.split(i)[-1]), 1)
        # else:
        #     print(f'Background images are ready under \'{background_folder}\'.')
        self.candidates = glob.glob(os.path.join(background_folder, '*'))
        self.zero_threshold = zero_threshold

    def get_background(self, shape):
        '''
        shape: size of the background
        '''
        # pdb.set_trace()
        if random.random() > self.zero_threshold:
            return np.zeros(shape, dtype='uint8')
        mode = random.choice([BACK_RANDOM, BACK_PURE, BACK_IMG])
        if mode == BACK_RANDOM:
            background = np.random.randint(0, 256, shape).astype('uint8')
        elif mode == BACK_PURE:
            background = np.zeros(shape, dtype='uint8')
            for i in range(3):
                background[:, :, i] = random.randint(0, 255)
        else:
            file = random.choice(self.candidates)
            background = np.asarray(Image.open(file))
            background = cv2.resize(background,
                                    (shape[1], shape[0]),
                                    cv2.INTER_AREA if any(np.asarray(shape) < np.asarray(background.shape))\
                                    else cv2.INTER_LINEAR)
        return background
