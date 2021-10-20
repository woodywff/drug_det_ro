import gc
import pdb
import time
from tools.utils import sec2hms
import torch
from math import pi



class Timer:
    def __init__(self):
        self.start_t = None

    def start(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.start_t = time.time()

    def end(self):
        torch.cuda.synchronize()
        res = {'consumed_t': sec2hms(time.time() - self.start_t),
               'consumed_mem_max': f'{round(torch.cuda.max_memory_allocated()/1e9, 4)} GB'}
        return res


def rotate_p_list(p_list, cx, cy, degree):
    '''
    pytorch version of tools.plot.rotate_p_list
    Get rotated x,y
    p_list: list of torch.Tensor; original points [[x0,y0], [x1,y1], ...]; absolute value;
    cx,cy: pivot coordinates; absolute value;
    degree: rotated angle
    RETURN: coordinates after rotation
    '''
    radian = degree * pi / 180
    p_list = torch.tensor(p_list, dtype=torch.float)
    p_c = torch.tensor([cx, cy], dtype=torch.float)
    return p_c + torch.matmul((p_list - p_c), (torch.tensor([[torch.cos(radian), torch.sin(radian)],
                                                             [-torch.sin(radian), torch.cos(radian)]])))
