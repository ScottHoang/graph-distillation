import os.path as osp
import random
import time

import numpy as np
import pyximport
import torch
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj


def init_dirs(args):
    timestr = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    logdir = osp.join(args.outdir, args.dataset, args.task, 'logs', timestr)
    configdir = osp.join(args.outdir, args.dataset, args.task, 'configs',
                         timestr)
    resultdir = osp.join(args.outdir, args.dataset, args.task, 'results',
                         timestr)
    return ((logdir, configdir, resultdir), timestr, args.outdir)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def pair_cosine_similarity(x, y, eps=1e-8):
    n1 = x.norm(p=2, dim=1, keepdim=True)
    n2 = y.norm(p=2, dim=1, keepdim=True)
    return x / n1.clamp(min=eps) @ (y / n2.clamp(min=eps)).t()


def update_args_dataset(data: pyg.data.Data, args: dict):
    """TODO: Docstring for update_args_dataset.

    :data: TODO
    :args: TODO
    :returns: TODO

    """
    args['num_nodes'] = data.x.size(0)
    args['num_features'] = data.x.size(1)
    args['num_classes'] = data.y.max() + 1
    return args
