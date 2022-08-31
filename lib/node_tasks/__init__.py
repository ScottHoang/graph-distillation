import argparse

from .classifier import MLPPredictor


def node_options(parents):
    parser = argparse.ArgumentParser(parents=parents, add_help=False)
    # for VGAE and GAE
    parser.add_argument('--smooth_label', action='store_true')
    parser.add_argument('--L_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--dropout', default=0.6, type=float)
    parser.add_argument('--disable_defaults', action='store_true')
    parser.add_argument('--patient', default=400, type=int)
    parser.add_argument('--fully_supervised', action='store_true')
    parser.add_argument("--density",
                        type=float,
                        default=0.1,
                        help="pooled graph density ")
    parser.add_argument("--teacher_network", type=str, default='')
    parser.add_argument("--pooling_network", type=str, default='')

    return parser


Tasks = {'mlp': MLPPredictor}
