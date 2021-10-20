import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import math
import json
import yaml
from collections import defaultdict
import cv2


def is_included(x, y, box_points):
    '''
    PREREQUISITE: box_points is a rectangle and points are arranged in order.
    Checi if a given point (x, y) is inside a rectangle defined by point list.
    x, y: point coordinates
    box_points: point list of a rectangle
    '''
    xs = [i[0] for i in box_points]
    xmin = min(xs)
    xmax = max(xs)
    if x > xmax or x < xmin:
        return False
    for i in range(2):
        p0 = box_points[i]
        p1 = box_points[i+1]
        p2 = box_points[i+2]
        # p3 = box_points[(i+3)%4]
        if p0[0] == p1[0]:
            return (p0[0] <= x <= p2[0] or p2[0] <= x <= p0[0]) and (p0[1] <= y <= p1[1] or p1[1] <= y <= p0[1])
        a = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b0 = p0[1] - a * p0[0]
        b1 = p2[1] - a * p2[0]
        if (y > a * x + b0 and y > a * x + b1) or (y < a * x + b0 and y < a * x + b1):
            return False
    return True

def ro_label_on_table(label, table, threshold):
    '''
    Update label on a certain area.
    label: ndarray; robndbox; [class, cx, cy, cw, ch, degree]; absolute values.
    table: list; [xmin, ymin, xmax, ymax]; absolute values.
    threshold: robndbox edge length smaller than threshold would be ignored.
    RETURN: processed label; ndarray.
    '''
    # pdb.set_trace()
    xmin, ymin, xmax, ymax = table
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    bound_corners = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    del_list = []
    for i in range(len(label)):
        rect = label[i][1:]
        rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
        old_center = rect[0]
        corners = cv2.boxPoints(rect).tolist()
        res = []
        for j in range(4):
            p0 = corners[j]
            p1 = corners[(j+1) % 4]
            for k in range(4):
                bc0 = bound_corners[k]
                bc1 = bound_corners[(k+1) % 4]
                res += get_pivot(p0, p1, bc0, bc1)
        res += corners
        # check bound corners
        for bc in bound_corners:
            if is_included(*bc, corners):
                res.append(bc)
        #     temp_x = sorted([p[0] for p in res if p[1] == bc[1]])
        #     temp_y = sorted([p[1] for p in res if p[0] == bc[0]])
        #     if temp_x and temp_y:
        #         if temp_x[0] <= bc[0] <= temp_x[-1] and temp_y[0] <= bc[1] <= temp_y[-1]:
        #             res.append(bc)
        res = [p for p in res if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]
        rect = get_robndbox_from_points(res)
        if rect is None or ((old_center[0] < xmin or old_center[0] > xmax or old_center[1] < ymin or old_center[1] > ymax) and \
                (rect[1][0] < threshold * w or rect[1][1] < threshold * h)):
            del_list.append(i)
            continue
        label[i][1] = (rect[0][0] - xmin) / w
        label[i][2] = (rect[0][1] - ymin) / h
        label[i][3] = rect[1][0] / w
        label[i][4] = rect[1][1] / h
        label[i][5] = rect[2]
    if del_list:
        label = np.delete(label, del_list, 0)
    return label


def get_pivot(p0, p1, p2, p3):
    '''
    Find the crossing point of two lines p0---p1, p2---p3
    p0,p1,p2,p3: list; e.g. [0,1]
    RETURN: point list; e.g. [[0,1], [2,3]]
    ATTENTION: return list may have redundant points.
    '''
    res = []

    if p0[0] == p1[0] == p2[0] == p3[0]:
        l0_ymin = min(p0[1], p1[1])
        l0_ymax = max(p0[1], p1[1])
        if (l0_ymin <= p2[1] <= l0_ymax) or (l0_ymin <= p3[1] <= l0_ymax):
            res += (sorted([p0, p1, p2, p3], key=lambda x: x[1])[1:3])
    elif p0[0] == p1[0] and p2[0] == p3[0]:
        pass
    elif p0[0] == p1[0]:
        if p2[0] <= p0[0] <= p3[0] or p3[0] <= p0[0] <= p2[0]:
            a1 = (p3[1] - p2[1]) / (p3[0] - p2[0])
            b1 = p2[1] - a1 * p2[0]
            y = a1 * p0[0] + b1
            if p0[1] <= y <= p1[1] or p1[1] <= y <= p0[1]:
                res.append([p0[0], y])
    elif p2[0] == p3[0]:
        if p0[0] <= p2[0] <= p1[0] or p1[0] <= p2[0] <= p0[0]:
            a0 = (p1[1] - p0[1]) / (p1[0] - p0[0])
            b0 = p0[1] - a0 * p0[0]
            y = a0 * p2[0] + b0
            if p2[1] <= y <= p3[1] or p3[1] <= y <= p2[1]:
                res.append([p2[0], y])
    else:
        a0 = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b0 = p0[1] - a0 * p0[0]
        a1 = (p3[1] - p2[1]) / (p3[0] - p2[0])
        b1 = p2[1] - a1 * p2[0]
        if a0 == a1 and b0 == b1:
            l0_xmin = min(p0[0], p1[0])
            l0_xmax = max(p0[0], p1[0])
            if (l0_xmin <= p2[0] <= l0_xmax) or (l0_xmin <= p3[0] <= l0_xmax):
                res += (sorted([p0, p1, p2, p3], key=lambda x: x[0])[1:3])
        elif a0 == a1:
            pass
        else:
            x = (b0 - b1) / (a1 - a0)
            y = a0 * x + b0
            if (p0[0] <= x <= p1[0] or p1[0] <= x <= p0[0]) and \
                    (p2[0] <= x <= p3[0] or p3[0] <= x <= p2[0]) and \
                    (p0[1] <= y <= p1[1] or p1[1] <= y <= p0[1]) and \
                    (p2[1] <= y <= p3[1] or p3[1] <= y <= p2[1]):
                res.append([x, y])
    return res

def get_robndbox_from_points(p_list, threshold=2):
    '''
    Return rotated bounding box of the polygon defined by given points
    p_list: ndarray or list.
    threshold: less than how many pixels do we ignore the robndbox
    ATTENTION: return robndbox as returned by minAreaRect in absolute values,
               or None if illegal or no robndbox is found.
    '''
    # pdb.set_trace()
    if len(p_list) <= 2:
        return
    rect = cv2.minAreaRect(np.asarray(p_list, dtype=int))
    rect = [list(rect[0]), list(rect[1]), rect[2]]
    if rect[1][0] < threshold or rect[1][1] < threshold:
        return
    if rect[2] == 90:
        rect[1][0],rect[1][1] = rect[1][1], rect[1][0]
        rect[2] = 0
    assert 0 <= rect[2] < 90, f'Unexpected rect angle degree {rect[2]}, points: {p_list}'
    return rect

def get_robndbox_from_img(alpha_img):
    '''
    Get rotated bounding box with cv2.minAreaRect from alpha channel.
    alpha_img: the alpha channel of a png image
    RETURN: the same as values returned by minAreaRect.
    '''
    contours = cv2.findContours(alpha_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].squeeze()
    return get_robndbox_from_points(contours)


def pad_rotate(img, degree):
    '''
    rotate and pad to include the whole rotated image
    img: ndarray
    degree: rotate angle clockwise
    RETURN: rotated img and transferring matrix
    '''
    # grab the dimensions of the image and then determine the center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -degree, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(img, M, (nW, nH))
    # perform the actual rotation and return the image
    return img, M


def parse_txt(txt_file):
    '''
    Transfer .txt annotations to ndarray format
    RETURN: ndarray
    ATTENTION: label class is gonna be in float format. Ratio values.
    '''
    with open(txt_file) as f:
        return np.asarray([i.strip().split() for i in f.readlines()], dtype=float)

class NameID():
    '''
    Maintain name2id and id2name in Dataset().
    * name is the annotation class name in .txt or .xml;
    * id is the label number saved in .h5.
    '''

    def __init__(self, name2id_path, id2name_path):
        '''
        name2id_path: name2id file path
        id2name_path: id2name file path
        '''
        self.name2id_path = name2id_path
        self.id2name_path = id2name_path
        self.name2id = json_read(name2id_path) if os.path.exists(name2id_path) else {}
        self.id2name = json_read(id2name_path) if os.path.exists(id2name_path) else {}
        assert len(self.name2id) == len(self.id2name), 'name2id doesn\'t fit with id2name'
        if self.id2name:
            self.id = max(list(map(int, self.id2name.keys()))) + 1
        else:
            self.id = 0
        return

    def update(self, name):
        '''
        What do we do when coming across an object in .txt or .xml.
        '''
        if self.name2id.get(name) is None:
            self.name2id[name] = str(self.id)
            self.id2name[self.id] = name
            self.id += 1
        return self.name2id[name]

    def reset(self):
        self.name2id = {}
        self.id2name = {}

    def save(self):
        '''
        Be careful before you do this!!!
        '''
        json_write(self.name2id, self.name2id_path)
        json_write(self.id2name, self.id2name_path)
        return

class Record:
    '''
    Record updated values.
    How to use it: record = Record(); record.update({key: value}); record.get_record()
    '''
    def __init__(self, round_digit=5):
        self.round_digit = round_digit
        self.record = defaultdict(list)
        return

    def __getattr__(self, item):
        if not self.record.get(item):
            raise Exception(f'There is no metric named {item} at this moment.')
        return np.mean(self.record[item]).round(self.round_digit)

    def get_record(self):
        return {i: getattr(self, i) for i in self.record.keys()}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.record[key].append(value)
        return

minmax = lambda x: (x - np.min(x))/(np.max(x)-np.min(x)) if np.max(x) != np.min(x) else np.zeros_like(x)

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def calc_param_size(params, in_byte=False):
    '''
    Show the memory cost of model.parameters, in MB.
    It works for float32 parameters.
    params: for tensorflow it's a list, for pytorch and paddlepaddle it's a generator.
    RETURN: Number of parameters if not in_byte, otherwise return the parameter memory consumption given param type is
            float32.
    '''
    res = np.sum(np.prod(p.shape) for p in params)
    return f'Parameter size: {res * 4e-6:.3f} MB' if in_byte else f'Number of parameters: {res * 1e-6:.3f} M'


def label_on_table(label, table, threshold=10):  
    '''
    Update label on a certain area.
    label: class, xmin, ymin, xmax, ymax
    table: xmin, ymin, xmax, ymax
    Absolute values of course.
    RETURN:
        cx, xy, cw, ch
        if None, ignore this label in the future, because the bbox is out of the boundary.
    '''
    if (min(label[3], table[2]) - max(label[1], table[0]) > threshold and \
       min(label[4], table[3]) - max(label[2], table[1]) > threshold) or \
            (table[0] <= label[1] <= table[2] and table[0] <= label[3] <= table[2] and \
            table[1] <= label[2] <= table[3] and table[1] <= label[4] <= table[3]):
        lxmin = max(label[1], table[0])
        lxmax = min(label[3], table[2])
        lymin = max(label[2], table[1])
        lymax = min(label[4], table[3])
        w = table[2] - table[0] + 1
        h = table[3] - table[1] + 1
        label[1] = ((lxmin + lxmax)/2 - table[0]) / w
        label[2] = ((lymin + lymax)/2 - table[1]) / h
        label[3] = (lxmax - lxmin + 1) / w
        label[4] = (lymax - lymin + 1) / h
        return np.asarray(label)


def xywh2xyxy(h, w, cx, cy, cw, ch):
    '''
    Change one bbox from yolo format to voc format.
    h: height of img
    w: width of img
    cx: coordinate-x of box center / width of img
    cy: coordinate-y of box center / height of img
    cw: width of box / width of img
    ch: height of box / height of img
    RETURN: coordinates of up-left and down-right corners.
    '''
    xmin = (cx - cw / 2) * w
    xmax = (cx + cw / 2) * w
    ymin = (cy - ch / 2) * h
    ymax = (cy + ch / 2) * h
    return xmin, ymin, xmax, ymax


def xyxy2xywh(h, w, xmin, ymin, xmax, ymax):
    '''
    Change one bbox from voc format to yolo format.
    h: height of img
    w: width of img
    xmin, ymin, xmax, ymax: coordinates of up-left and down-right corners.
    RETURN:
        cx: coordinate-x of box center / width of img
        cy: coordinate-y of box center / height of img
        cw: width of box / width of img
        ch: height of box / height of img
    '''
    cx = (xmax + xmin) / 2 / w
    cy = (ymax + ymin) / 2 / h
    cw = (xmax - xmin) / w
    ch = (ymax - ymin) / h
    return cx, cy, cw, ch

def label_xywh2xyxy(label, h, w, inplace=True):
    '''
    Change targets from yolo format to voc format
    label: [num of objs, 5] class, cx, cy, cw, ch, ratio values.
    h: height of img
    w: width of img
    inplace: if False, copy label.
    Return label in absolute values.
    '''
    new_label = label if inplace else np.copy(label)
    for i in range(len(label)):
        new_label[i, 1:] = xywh2xyxy(h, w, *label[i, 1:])
    return new_label

def label_xyxy2xywh(label, h, w, inplace=True):
    '''
    Change targets from voc format to yolo format
    label: [num of objs, 5] class, xmin, ymin, xmax, ymax
    h: height of img
    w: width of img
    inplace: if False, copy label.
    '''
    new_label = label if inplace else np.copy(label)
    for i in range(len(label)):
        new_label[i, 1:] = xyxy2xywh(h, w, *label[i, 1:])
    return new_label

def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))
    return

def json_read(file):
    with open(file) as f:
        return json.load(f)
    
def json_write(dic, file):
    with open(file, 'w') as f:
        json.dump(dic, f, ensure_ascii=False)
    return

def yaml_write(dic, file):
    with open(file, 'w') as f:
        yaml.dump(dic, f)
    return
    
def yaml_read(file):    
    with open(file) as f:
#         return yaml.load(f, Loader=yaml.FullLoader)
        return yaml.load(f, Loader=yaml.Loader)

def sec2hms(t, trivial=False):
    '''
    Transfer seconds to format hour:min:sec:msec:usec:nsec
    t: time in second, float
    RETURN: str
    '''
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    if not trivial:
        s = round(t % 60, 6)
        return f'{h} hours, {m} mins, {s} secs'
    s = int(t % 60)
    f = t - int(t)
    ms = int(f * 1000)
    us = int(f * 1e6 - ms * 1e3)
    ns = round(f * 1e9 - ms * 1e6 - us * 1e3)
    
    return f'{h} hours, {m} mins, {s} secs, {ms} m-secs, {us} u-secs, {ns} n-secs'

def mkdirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return

def plot_fig(img,figsize=(10,5)):
    '''
    Image show and plot histogram.
    '''
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.hist(img.reshape(-1))
    plt.tight_layout()
    return

