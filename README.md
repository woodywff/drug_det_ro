# Towards Toxic and Narcotic Medication Detection with Rotated Object Detector

## Introduction
This is the source code of article: [Towards Toxic and Narcotic Medication Detection with Rotated Object Detector](https://arxiv.org/abs/2110.09777)  
The orgnization of this repo looks like this:
```
.
├── configs 
│   ├── cfg_ro.yml # main config file for rotated yolo-v5
│   ├── cfg.yml    # main config file for yolo-v5
│   ├── model_pt   # model config files
│   │   ├── yolov5s_ro.yml
│   │   └── yolov5s.yml
│   ├── nms        # config file for nms
│   │   └── extra_filter.json
│   └── pipeline   # config file for data augmentation
│       └── aug_cfg.yml
├── pipeline       # Analogy to Dataset in Pytorch
│   ├── augment.py 
│   └── dataset.py
├── pt             # Pytorch specific implementation
│   ├── common.py  # DL basic modules
│   ├── loss.py    # loss function ralated for yolo-v5
│   ├── loss_ro.py # loss function ralated for rotated yolo-v5
│   ├── metric.py  # Evaluation ralated
│   ├── server.py  # Main classes for training validation and inference
│   ├── utils.py   # Pytorch specific utilities
│   ├── yolo.py    # Model classes of yolo-v5
│   ├── yolo_ro.py # Model classes of rotated yolo-v5
│   └── log        
│       └── ...    # Where do we save the trained parameters (.pt)
├── tools          # Helper functions
│   ├── colormap.py
│   ├── compress.py
│   ├── const.py
│   ├── plot.py
│   └── utils.py   # Framework independent utilities
├── plot4latex.ipynb # How do we get the figures in the article
├── train.py       # Command for training 
└── infer.py       # Set up an inference http server
```
## How to Get Started
### Prerequisite
Class id in .txt label file has already been transfered to the index number we finally use in training and inference. There should be a `yolo_label_id2name.json` file saving the mapping.    
All the tunable arguments are listed in `configs/cfg_ro.yml` for rotated yolo-v5 and `configs/cfg.yml` for yolo-v5. It's almost self-explainable, feel free to play with it please.

### Training
For rotated yolo-v5:  
`python train.py --cfg=configs/cfg_ro.yml`

For yolo-v5:  
`python train.py --cfg=configs/cfg.yml`

### Inference
For rotated yolo-v5:  
`python infer.py --cfg=configs/cfg_ro.yml`

For yolo-v5:  
`python infer.py --cfg=configs/cfg.yml`

This would start up an inference http server with the best-shot trained parameters.

## Development Environment
RTX 3060 (12GB GPU Memory)
CUDA 11.2
Python 3.8
python packages:
`requirements.txt`


## Acknowledgment
This work refers a lot to [ultralytics/yolov5](https://github.com/ultralytics/yolov5) and [BossZard/rotation-yolov5](https://github.com/BossZard/rotation-yolov5). We deeply appreciate their contributions to the community.


## Citation

Bibtex

```
@article{adam,
  title={Towards Toxic and Narcotic Medication Detection with Rotated Object Detector},
  author={Peng, Jiao and Wang, Feifan and Fu, Zhongqiang and Hu, Yiying and Chen, Zichen and Zhou, Xinghan and Wang, Lijun},
  journal={arXiv preprint arXiv:2110.09777},
  year={2021},
  url={https://arxiv.org/abs/2110.09777}
}

```