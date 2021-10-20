import numpy as np
import torch
import os
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict, OrderedDict
import shutil
from tools.const import *
from tools.utils import yaml_read, print_red, mkdirs, calc_param_size, Record, minmax, json_read
from pipeline.dataset import Dataset, Generator
from pt.utils import Timer
from pt.metric import Metric, NMS
from pipeline.augment import PadResize
from datetime import datetime


class Base:
    '''
    Parent class for Server
    '''
    def __init__(self, cfg_path, infer_only=False):
        '''
        cfg_path: config file cfg.yml path.
        infer_only: if True, only for inference.
        '''
        self.cfg_path = cfg_path
        self._init_cfg()
        self._init_log()
        self._init_device()
        if not infer_only:
            self._init_dataset()
            self.timer = Timer()

    def _init_cfg(self):
        '''
        Parse configurations (cfg.yml + argparse).
        '''
        self.cfg = yaml_read(self.cfg_path)
        self.model_cfg = yaml_read(self.cfg['model_cfg']['pt'])
        if self.cfg['is_ro']:
            self.model_cfg['n_degree'] = self.cfg['n_degree']
            print('Init config: is_ro = True,', 'n_degree =', self.cfg['n_degree'])


    def _init_log(self):
        '''
        Get ready folders for saving and log.
        '''
        if self.cfg['is_ro']:
            self.log_path = os.path.join(self.cfg['log_path']['pt'], 'degree_{}'.format(self.cfg['n_degree']))
        else:
            self.log_path = self.cfg['log_path']['pt']
        mkdirs(self.log_path)
        self.train_last = os.path.join(self.log_path, 'last.pt')
        self.train_best = os.path.join(self.log_path, 'best.pt')


    def _init_device(self):
        '''
        Check which device to be deployed on, GPU or CPU.
        '''
        seed = self.cfg['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed(seed)
        else:
            print_red('No gpu devices available! We will use cpu')
            self.device = torch.device('cpu')
        return

    def _init_dataset(self):
        '''
        Dataset initialization.
        '''
        self.dataset = Dataset(img_folder=self.cfg['dataset']['img_folder'],
                               label_folder=self.cfg['dataset']['label_folder'],
                               dataset_folder=self.cfg['dataset']['dataset_folder'])
        split_place = int(self.cfg['split_prop'] * len(self.dataset.ids))
        self.train_generator = Generator(h5_path=self.dataset.h5,
                                         ids=self.dataset.ids if self.cfg['train']['train_all']
                                         else self.dataset.ids[:split_place],
                                         batch_size=self.cfg['train']['batch_size'],
                                         shape=self.cfg['shape'],
                                         aug_cfg=self.cfg['aug_cfg'],
                                         is_ro=self.cfg['is_ro'])
        self.val_generator = Generator(h5_path=self.dataset.h5,
                                       ids=self.dataset.ids if self.cfg['train']['train_all']
                                       else self.dataset.ids[split_place:],
                                       batch_size=self.cfg['val']['batch_size'],
                                       shape=self.cfg['shape'],
                                       is_ro=self.cfg['is_ro'])
        return


class Server(Base):
    '''
    Controller of training, validation and test process.
    '''
    def __init__(self, cfg_path, infer_only=False, infer_model=None):
        '''
        cfg_path: config file cfg.yml path.
        infer_only: if True, only for inference.
        infer_model: if given, use certain trained model.
        '''
        super().__init__(cfg_path, infer_only)
        self.infer_only = infer_only
        self._init_model()
        if infer_only:
            self.infer_model = infer_model
            self._infer_init()
        else:
            self._check_resume(reset_lr=self.cfg['train']['reset_lr'])

    def _init_model(self):
        self.n_classes = len(json_read(self.cfg['dataset']['yolo_label_id2name']))
        if self.cfg['is_ro']:
            from pt.yolo_ro import RoYolov5
            from pt.loss_ro import ComputeLoss
            self.model = RoYolov5(self.model_cfg,
                                  nc=self.n_classes,
                                  infer_only=self.infer_only).to(self.device)
        else:
            from pt.yolo import Yolov5
            from pt.loss import ComputeLoss
            self.model = Yolov5(self.model_cfg,
                                nc=self.n_classes,
                                infer_only=self.infer_only).to(self.device)
        if not self.infer_only:
            print(calc_param_size(self.model.parameters()))
            self.loss = ComputeLoss(self.model)
            self._init_optim()
        else:
            self.loss = None # for evaluation

    def _init_optim(self):
        self.optim = Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optim, verbose=True, factor=0.5)
        return

    def _check_resume(self, reset_lr=False):
        '''
        Restore saved model, parameters and statistics.
        reset_lr: if True, reset learning rate to the original, otherwise, load the saved.
        '''
        if os.path.exists(self.train_last):
            print('Continue training from last stop point...')
            state_dicts = torch.load(self.train_last, map_location=self.device)
            self.epoch = state_dicts['epoch'] + 1
            self.history = state_dicts['history']
            self._load_model(state_dicts['model_param'])
            print(f'Loaded pre-trained model from {self.train_last}.')
            if not reset_lr:
                self.optim.load_state_dict(state_dicts['optim'])
                self.scheduler.load_state_dict(state_dicts['scheduler'])
            self.best_loss = state_dicts['best_loss'] if os.path.exists(self.train_best) else float('inf')
        else:
            print('Training from scratch...')
            pre_trained_path = self.cfg['pre_trained']
            if pre_trained_path:
                state_dicts = torch.load(pre_trained_path, map_location=self.device)
                self._load_model(state_dicts['model_param'])
                print(f'Loaded pre-trained model from {pre_trained_path}.')
            self.epoch = 0
            self.history = defaultdict(list)
            self.best_loss = float('inf')

    def _load_model(self, model_param):
        '''
        Load param for those model having the same dimension as the saved model.
        model_param:  parameter dict.
        '''
        new_param = self.model.state_dict()
        loadable_dict = {key: value for key, value in model_param.items() if
                         new_param[key].shape == value.shape}
        new_param.update(loadable_dict)
        self.model.load_state_dict(new_param)

    def _infer_init(self):
        '''
        Load infer model from given path or load the default self.train_best
        '''
        state_dicts = torch.load(self.infer_model or self.train_best, map_location=self.device)
        self.model.load_state_dict(state_dicts['model_param'])
        self.model.eval()
        self.nms = NMS(self.n_classes,
                       is_ro=self.cfg['is_ro'],
                       n_degree=self.model.cfg.get('n_degree'),
                       **self.cfg['nms'])

    def train(self, verbose=True):
        '''
        Training process
        verbose: if True, print one pbar for each single_epoch, otherwise only one pbar for the whole training.
        '''
        n_epochs = self.cfg['train']['epochs']
        n_aug_epochs = self.cfg['train']['aug_epochs']
        patience = 0
        old_loss = float('inf')

        if verbose:
            pbar = range(n_aug_epochs)
        else:
            pbar = tqdm(range(n_aug_epochs), desc = 'Training')
        for aug_epoch in pbar:
            postdict = OrderedDict({'AugEpoch': aug_epoch,
                                   'Epoch in all': self.epoch})
            is_best = False
            self.timer.start()
            res_training = self.single_training_epoch(self.train_generator, verbose=verbose, aug_epoch=aug_epoch)
            postdict.update(res_training)
            postdict.update(self.timer.end())

            res_val = self.single_val_epoch(self.val_generator, verbose=verbose, aug_epoch=aug_epoch)
            postdict.update({f'val_{key}': value for key, value in res_val.items()})
            if not verbose:
                pbar.set_postfix(postdict)
            else:
                print('Timestamp:', datetime.now().strftime('%Y-%m-%d, %H:%M:%S'))
                for key, value in postdict.items():
                    if key == 'val_mAP' and self.epoch % self.cfg['metric_step'] != 0:
                        continue
                    print(f'{key}: {value}')

            for key, value in postdict.items():
                self.history[key].append(value)

            loss = res_val['loss']
            if loss < self.best_loss and aug_epoch > self.cfg['train']['warmup_epochs']:
                is_best = True
                self.best_loss = loss

            # Save what the current epoch ends up with.
            state_dicts = {
                'epoch': self.epoch,
                'history': self.history,
                'model_param': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_loss': self.best_loss
            }
            torch.save(state_dicts, self.train_last)

            if is_best:
                shutil.copy(self.train_last, self.train_best)

            if self.epoch % self.cfg['metric_step'] == 0:
                shutil.copy(self.train_last, os.path.join(self.log_path,
                                                          '{}.{}'.format(self.epoch, self.train_last.split('.')[-1])))

            if loss >= old_loss:
                patience += 1
                if patience == self.cfg['train']['patience']:
                    print(f'We\'ve seen no decrease of loss value in the past {patience} epochs. EARLY STOP.')
                    break
            else:
                old_loss = loss
                patience = 0

            self.epoch += 1
            if self.epoch > n_epochs:
                return
        torch.cuda.empty_cache()
        return

    def single_training_epoch(self, generator, verbose=False, aug_epoch=0):
        '''
        Single epoch for training process
        generator: instance of class Generator.
        verbose: if True, print one pbar for each single_epoch, otherwise only one pbar for the whole training.
        return: dict saving loss values.
        '''

        self.model.train()
        n_steps = generator.steps_per_epoch
        record = Record()
        if verbose:
            pbar = tqdm(generator.epoch(),
                        total=n_steps,
                        desc=f'Training | AugEpoch {aug_epoch} | Epoch in all {self.epoch}')
        else:
            pbar = generator.epoch()
        scaler = torch.cuda.amp.GradScaler() if self.cfg['train']['amp'] else None

        for x, y in pbar:
            x = torch.as_tensor(x, device=self.device, dtype=torch.float)
            y = torch.as_tensor(y, device=self.device, dtype=torch.float)
            self.optim.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    res_loss = self.loss(pred, y)
                scaler.scale(res_loss['loss']).backward()
                scaler.step(self.optim)
                scaler.update()
            else:
                pred = self.model(x)
                res_loss = self.loss(pred, y)
                res_loss['loss'].backward()
                self.optim.step()
            record.update(**{key: value.item() for key, value in res_loss.items()})

            if verbose:
                pbar.set_postfix(record.get_record())
        self.scheduler.step(res_loss['loss'])
        return record.get_record()

    @torch.no_grad()
    def single_val_epoch(self, generator, verbose=False, aug_epoch=0):
        '''
        Single epoch for val process
        generator: instance of class Generator.
        verbose: if True, print one pbar for each single_epoch, otherwise only one pbar for the whole training.
        aug_epoch: number of epoch in the current training process.
        RETURN: dict saving loss values.
        '''

        self.model.eval()
        n_steps = generator.steps_per_epoch
        record = Record()
        metric = Metric(n_cls=self.n_classes,
                        is_ro=self.cfg['is_ro'],
                        n_degree=self.model.cfg.get('n_degree'),
                        cfg=self.cfg['nms'])
        if verbose:
            pbar = tqdm(generator.epoch(),
                        total=n_steps,
                        desc='Evaluation' if self.infer_only else f'Val | AugEpoch {aug_epoch} | Epoch {self.epoch}')
        else:
            pbar = generator.epoch()

        for x, y in pbar:
            x = torch.as_tensor(x, device=self.device, dtype=torch.float)
            y = torch.as_tensor(y, device=self.device, dtype=torch.float)
            pred = self.model(x)
            res_loss = self.loss(pred, y) if self.loss else {} # empty is for evaluation
            record.update(**{key: value.item() for key, value in res_loss.items()})
            if self.infer_only or self.epoch % self.cfg['metric_step'] == 0:
                pred = self.model.post_process(pred)
                metric.update(pred, y)
            if verbose:
                pbar.set_postfix(record.get_record())
        if self.infer_only or self.epoch % self.cfg['metric_step'] == 0:
            metric.calc_metrics()
            record.update(mAP=metric.mAP)
        return metric if self.infer_only else record.get_record()

    @torch.no_grad()
    def single_infer(self, img_path):
        '''
        Inference on one image.
        img_path: input image file path.
        RETURN: ndarray; Since it's single inference, the returned pred shape is [num obj, 7 or 6];
                7 for robndbox: cx, cy, cw, ch, degree, conf, class
                6 for bndbox:   cx, cy, cw, ch, conf, class
        '''
        img = np.asarray(Image.open(img_path))
        assert np.all(img >= 0) and np.all(img <= 255), f'We need 0 ~ 255 images while {img_path} is not.'
        h, w = self.cfg['shape']
        img, _, pad_list = PadResize(h=h, w=w, sym=True, return_pad=True)(img)

        img = minmax(img/255.0)
        img = img.transpose(2,0,1)[None,...]
        x = torch.as_tensor(img, device=self.device, dtype=torch.float)
        pred = self.model(x)
        pred = self.model.post_process(pred)
        pred = self.nms(pred)[0]
        if isinstance(pred, int):
            return np.asarray([])
        else:
            # return cx, cy, cw, ch for the original size
            top, bottom, left, right = pad_list
            pred[:, 0] = (pred[:, 0] * w - left) / (w - left - right)
            pred[:, 1] = (pred[:, 1] * h - top) / (h - top - bottom)
            pred[:, 2] = pred[:, 2] * w / (w - left - right)
            pred[:, 3] = pred[:, 3] * h / (h - top - bottom)
            return pred

    @torch.no_grad()
    def evaluate(self, img_folder, label_folder, dataset_folder, temp_id=None):
        '''
        Evaluation on certain dataset.
        img_folder: where are the images.
        label_folder: where are the annotations.
        dataset_folder: target folder in which to save all the generated files (dataset.h5).
        '''
        dataset = Dataset(img_folder, label_folder, dataset_folder)
        if temp_id:
            single_id = [temp_id]
            print(single_id)
            eval_generator = Generator(h5_path=dataset.h5, ids=single_id)
        else:
            eval_generator = Generator(h5_path=dataset.h5, ids=dataset.ids)
        return self.single_val_epoch(eval_generator, verbose=True)

