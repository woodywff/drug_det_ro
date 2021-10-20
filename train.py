'''
Training process
'''

import argparse
from pt.server import Server


def parse_args():
    '''
    This inference gets started as an http server.
    '''
    parser = argparse.ArgumentParser(description='Ex: python train.py --cfg=configs/cfg_ro.yml')
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Config yaml file path.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FLAGS = parse_args()
    server = Server(cfg_path=FLAGS.cfg)
    server.train()
