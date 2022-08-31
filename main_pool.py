import argparse
import collections
import json
import os
import os.path as osp
import statistics as stats
import time

import pandas as pd
import torch
import torchmetrics
import tqdm
from torch.utils.tensorboard import SummaryWriter

from lib.pooling_tasks import pooling_options
from lib.pooling_tasks import Tasks as pool_tasks
from lib.utils import init_dirs
from lib.utils import set_seed


def default_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str, default='Cora')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--outdir", default='pooling')
    parser.add_argument("--task", type=str)
    parser.add_argument("--N-exp", default=1, type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", default=100, type=int)
    return parser


def main():
    parser = pooling_options([default_parser()])
    args = parser.parse_args()

    pool_fn = pool_tasks[args.task]

    dirs, time, basedir = init_dirs(args)  #, task_type=args.task)
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    logdir, configdir, resultdir = dirs
    with open(osp.join(configdir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    args.logdir = logdir
    args.config = configdir
    args.resultdir = resultdir

    writer = SummaryWriter(logdir)

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    dp = collections.defaultdict(list)
    try:
        for seed in range(args.N_exp):
            set_seed(seed)
            task = pool_fn(args, seed)
            results = task.fit(writer)
            task.save()
            dp['task'].append(args.task)
            for k, v in results.items():
                dp[k].append(v)
    except KeyboardInterrupt:
        print("Keyboard interrupted! saving results")

    dp = pd.DataFrame.from_dict(dp)
    dp.to_csv(osp.join(resultdir, 'results.csv'))

    writer.close()


if __name__ == "__main__":
    main()
