import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
from PIL import Image
import numpy as np
from tools.utils import json_read
import pdb


def npy2txt(labels, save_file=None):
    '''
    Transfer an ndarray to .txt format for yolo_bbox_check need.
    RETURN list of bndbox or robndbox
    ATTENTION: res is a list rather than ndarray because of the inconsistency in item format 
               (the 1st one is int, the others are float)
    '''
    res = [' '.join([str(int(i[j]) if j == 0 else i[j]) for j in range(len(i))]) for i in labels]
    if save_file:
        with open(save_file, 'w') as f:
            f.writelines(res)
    return res


def yolo_bbox_check(img, txt=None, ro_txt=None, 
                    axis_off=False, save=True, save_path='test.png', 
                   id2name=None, yolo_label_id2name=None,
                   fontsize=15, c_text='r', c_bndbox='r', c_robndbox='cyan',
                    trivial=True, ax=None):
    '''
    img_file: image file path or ndarray; [height, width, channel], RGB.
    txt_file: rectangle label .txt file path or ndarray; [class, cx, cy, cw, ch]; ratio value.
    ro_txt_file: rotated rectangle label .txt file or ndarray; [class, cx, cy, cw, ch, degree]; ratio value.
    axis_off: if True, turn off axis on plot.
    trivial: if True, show the annotations.
    ax: if given, draw on the given figure.
    '''
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    if isinstance(img, str):
        img = np.asarray(Image.open(img))
    h,w = img.shape[:2]
    ax.imshow(img)
    if txt is not None:
        if isinstance(txt, str):
            with open(txt) as f:
                labels = f.readlines()
        else:
            labels = txt
        for label in labels:
            label = label.strip().split()
            cx = float(label[1])*w
            cy = float(label[2])*h
            cw = float(label[3])*w
            ch = float(label[4])*h
            plot_rec_xywh(ax, cx, cy, cw, ch, c=c_bndbox)
            if trivial:
                ax.text(cx, cy, json_read(id2name)[json_read(yolo_label_id2name)[label[0]]], 
                        fontsize=fontsize, color=c_text)
    if ro_txt is not None:
        if isinstance(ro_txt, str):
            with open(ro_txt) as f:
                labels = f.readlines()
        else:
            labels = ro_txt
        for label in labels:
            label = label.strip().split()
            cx = float(label[1]) * w
            cy = float(label[2]) * h
            cw = float(label[3]) * w
            ch = float(label[4]) * h
            degree = float(label[5])
            plot_rotated_minAreaRect(ax, cx, cy, cw, ch, degree, c=c_robndbox)
            if trivial:
                ax.text(cx, cy, json_read(id2name)[json_read(yolo_label_id2name)[label[0]]], 
                        fontsize=fontsize, color=c_text)
    if axis_off:
        plt.axis('off')
    if save:
        plt.savefig(save_path)
    return ax
        

def plot_rec_xywh(ax, x, y, w, h, c='y', ls='-', lw=2):
    '''
    Plot rectangle.
    xywh: abs value of center and rec edge length.
    '''
    xmin = (x - w/2)
    ymin = (y - h/2)
    xmax = (x + w/2)
    ymax = (y + h/2)
    plot_rec_xyxy(ax, xmin, ymin, xmax, ymax, c, ls, lw)
    return ax


def plot_rec_xyxy(ax, xmin, ymin, xmax, ymax, c='y', ls='-', lw=2):
    '''
    Plot rectangle.
    xmin, ymin, xmax, ymax: abs value of 4 points.
    '''
    draw_line = lambda x0, x1, y0, y1: ax.plot([x0, x1], [y0, y1], color=c, linewidth=lw, linestyle=ls)
    draw_line(xmin, xmax, ymin, ymin)
    draw_line(xmin, xmax, ymax, ymax)
    draw_line(xmin, xmin, ymin, ymax)
    draw_line(xmax, xmax, ymin, ymax)
    return ax        


def plot_polygon(ax, p_list, c='r', ls='-', lw=2):
    '''
    Plot polygon.
    ax: matplot ax
    p_list: point list, [[x0,y0],[x1,y1],...]
    c: color
    ls: line style
    lw: line width
    '''
    draw_line = lambda p0, p1: ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, linewidth=lw, linestyle=ls)
    n = len(p_list)
    for i in range(n):
        draw_line(p_list[i], p_list[(i+1)%n])
    return ax


def plot_rotated_minAreaRect(ax, cx, cy, cw, ch, degree, c='r', ls='-', lw=3):
    '''
    PLot rotated rectangle returned by cv2.minAreaRect
    ax: matplot ax
    cx,cy,cw,ch,degree: format as values returned by cv2.minAreaRect
    c: color
    ls: line style
    lw: line width
    RETURN: list of points
    '''
    p_list = [[cx - cw/2, cy - ch/2], [cx + cw/2, cy - ch/2], [cx + cw/2, cy + ch/2], [cx - cw/2, cy + ch/2]]
    rotated_p_list = rotate_p_list(p_list, cx, cy, degree)
    plot_polygon(ax, rotated_p_list, c, ls, lw)
    return rotated_p_list


def rotate_p_list(p_list, cx, cy, degree):
    '''
    Get rotated x,y
    p_list: ndarray; original points [[x0,y0], [x1,y1], ...]; absolute value;
    cx,cy: pivot coordinates; absolute value;
    degree: rotated angle 
    RETURN: coordinates after rotation
    '''
    radian = degree * np.pi / 180
    p_list = np.asarray(p_list)
    p_c = np.asarray([cx, cy])
    return p_c + (p_list - p_c).dot(np.asarray([[np.cos(radian), np.sin(radian)],
                                                [-np.sin(radian), np.cos(radian)]]))
