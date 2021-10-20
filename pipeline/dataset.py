import h5py
import numpy as np
import glob
import os
from tools.const import *
from tools.utils import mkdirs, minmax, yaml_read, parse_txt, print_red
from PIL import Image
from random import shuffle
from pipeline.augment import (Augment, HsvAug, Flip, Mosaic, PadRotate,
                              Mov, Shrink, Crop, Background, PadResize)

import pdb

'''
The pipeline provides ndarray rather than framework-specific formats.
'''
class Generator():
    '''
    Trainging and validation dataset generator
    '''
    def __init__(self, h5_path, ids, batch_size=1, shape=(640, 640),
                 aug_cfg=None, for_infer=False, do_norm=True, is_ro=True):
        '''
        h5_path: dataset.h5 file path.
        ids: id list, including the candidate samples we are gonna use.
        batch_size: as name suggests.
        shape: target image shape. list or tuple. (height, width)
        aug_cfg: augmentation config file path.
        for_infer: if True, for inference process.
        do_norm: if True, divide 255 and minmax() at the end of self._append()
        is_ro: if True, for rotated bounding boxes.
        '''
        self.h5_path = h5_path
        self.ids = ids
        self.batch_size = batch_size
        self.h, self.w = shape
        self.aug_cfg = aug_cfg
        self.for_infer = for_infer
        self.do_norm = do_norm
        self.is_ro = is_ro

        self.init_augment()
        self.steps_per_epoch = int(np.ceil(len(self.ids)/self.batch_size))
        
    def init_augment(self):
        '''
        Initialize an Augment instance.
        We do the mosaic augmentation first, so you only need to take care of the shape once.
        For each __call__ there would be different background if backgrould is not None, otherwise, 
        there is only one choice left -- pure black background.
        If there is an augment config .yml file, load it,
        otherwise, reset the aug_list to initialize with.
        '''
        self.background = Background()
        if self.aug_cfg is None:
            self.aug = Augment([PadResize(h=self.h, w=self.w, sym=True)])
        else:
            self.aug_cfg = yaml_read(self.aug_cfg)
            aug_list = self._parse_aug_list(self.aug_cfg['aug_list'])
            self.aug = Augment(aug_list)

    def _parse_aug_list(self, str_aug_list):
        '''
        Parse the aug_list, from class names to instances.
        str_aug_list: list of class names in str format.
        '''
        aug_list = []    
        for aug in str_aug_list:
            if isinstance(aug, list):
                aug_list.append(self._parse_aug_list(aug))
            elif aug == 'Mosaic': # Mosaic have to be coming first.
                aug_list.append(eval(aug)(h5_path=self.h5_path,
                                          id_list=self.ids,
                                          background=self.background,
                                          h=self.h,
                                          w=self.w,
                                          is_ro=self.is_ro,
                                          threshold=self.aug_cfg['threshold'],
                                          **self.aug_cfg[aug]))
            elif aug == 'HsvAug':
                aug_list.append(eval(aug)(**self.aug_cfg[aug]))
            elif aug == 'Flip':
                aug_list.append(eval(aug)(is_ro=self.is_ro,
                                          **self.aug_cfg[aug]))
            elif aug in ['Crop', 'Mov']:
                aug_list.append(eval(aug)(background=self.background,
                                          threshold=self.aug_cfg['threshold'],
                                          is_ro=self.is_ro,
                                          **self.aug_cfg[aug]))
            elif aug == 'PadRotate':
                if self.is_ro:
                    aug_list.append(eval(aug)(background=self.background))
                else:
                    continue
            elif aug == 'Shrink':
                aug_list.append(eval(aug)(background=self.background,
                                          **self.aug_cfg[aug]))
            else:
                raise NotImplementedError('Unknown augmentation class.')
        return aug_list
        
    def epoch(self):
        '''
        Yield data for one epoch.
        '''
        ids = self.ids.copy()
        if not self.for_infer:
            shuffle(ids)
        x = []
        y = []
        belong_to = 0 # which index of img in one batch does the label belong to.
        while ids:
            i = ids.pop()
            self._append(x, y, i, belong_to)
            belong_to += 1
            if len(x) == self.batch_size or not ids:
                yield self._feed(x, y)
                x = []
                y = []
                belong_to = 0
        return
    
    def _append(self, x, y, i, belong_to):
        '''
        Append one sample on the batch. This is also where we do the augmentation.
        Dataset specific. 
        This one is for object detection projects.
        x: input list (640, 640, 3)
        y: label list
        i: id in dataset.h5.
        belong_to: which index of img in one batch does the label belong to.
        '''
        with h5py.File(self.h5_path, 'r') as f:
            img = np.asarray(Image.open(f[i].attrs['img']))
            label = np.asarray(f[i]['label'])
            img, label = self.aug(img, label)
            label_length = 6 if self.is_ro else 5
            if len(label) == 0:
                if label.shape == (0,):
                    label = np.repeat(label[:, None], label_length, axis=-1)
                if label.shape != (0, label_length):
                    raise Exception('label shape != (0, 6 or 5)')
            label = np.column_stack([np.ones((len(label), 1)) * belong_to, label])
            if self.do_norm:
                img = minmax(img/255.0)
            x.append(img)
            y.append(label)
        return

    @staticmethod
    def _feed(x, y):
        '''
        This is like collate function in the other frameworks.
        x: input list, (640, 640, 3).
        y: label list, (batch size, 6), index in batch, class, cx, cy, cw, ch.
        '''
        x = np.asarray(x).transpose((0, 3, 1, 2))
        y = np.row_stack(y)
        return np.ascontiguousarray(x), np.ascontiguousarray(y)


class Dataset:
    '''
    Dataset creator and maintainer.
    It only supports yolo dataset in this repository.
    * Generated files: [dataset_folder]/dataset.h5 which saves the img path and labels in ndarray.
    '''
    def __init__(self, img_folder, label_folder, dataset_folder, overwrite=False,
                 ignore_grayscale=True):
        '''
        img_folder: where are the images.
        label_folder: where are the annotations.
        dataset_folder: target folder in which to save all the generated files.
        overwrite: if True, generate and overwrite the existing .h5 file.
        ignore_grayscale: if True, get rid of grayscale images.
        '''
        self.img_folder = img_folder[:-1] if img_folder[-1] == '/' else img_folder
        self.label_folder = label_folder[:-1] if label_folder[-1] == '/' else label_folder
        self.ignore_grayscale = ignore_grayscale

        mkdirs(dataset_folder)
        self.h5 = os.path.join(dataset_folder, H5FILE)

        if os.path.exists(self.h5) and not overwrite:
            print(f'{self.h5} exists already.')
        else:
            self.gen_h5()
        self.ids = self.get_ids()
        return
    
    def get_ids(self):
        '''
        Return id list in .h5
        '''
        with h5py.File(self.h5, 'r') as f:
            return list(f.keys())

    def gen_h5(self):
        '''
        Generate the .h5 file.
        Check image format and value and channels.
        '''
        img_list = glob.glob(os.path.join(self.img_folder, '*'))
        shuffle(img_list)
        with h5py.File(self.h5, 'w') as f:
            for i, img_file in enumerate(tqdm(img_list, desc = f'Creating {self.h5}')):
                assert img_file[-3:] in SUPPORTED_IMG_FORMAT, f'Unsupported image format: {img_file}'
                filename = os.path.basename(img_file).split('.')[0]
                label_file = os.path.join(self.label_folder, filename + '.txt')

                img = np.asarray(Image.open(img_file))
                assert np.all(img >= 0) and np.all(img <= 255), f'We need 0 ~ 255 images while {img_file} is not.'
                assert img.dtype in [int,'uint8'], f'{img_file}.dtype is not int.'
                if len(img.shape) == 2:
                    if self.ignore_grayscale:
                        continue
                    img = img[..., None]
                    img = img.repeat(3, axis=2)
                    Image.fromarray(img).save(img_file)
                assert img.shape[2] == 3, f'Image Format Exception ! The shape of {img_file} is {img.shape}.'
                label = parse_txt(label_file)
                f.create_group(filename)
                f[filename].attrs['img'] = img_file
                f[filename]['label'] = label
                if DEBUG_FLAG:
                    if i > 100:
                        break
            f.attrs['img_folder'] = self.img_folder
            f.attrs['label_folder'] = self.label_folder
        return

