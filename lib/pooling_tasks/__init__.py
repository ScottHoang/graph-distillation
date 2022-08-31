import argparse

from .diffpool import DiffPoolPredictor


def pooling_options(parents):
    parser = argparse.ArgumentParser(parents=parents, add_help=False)
    # for VGAE and GAE
    parser.add_argument('--smooth_label', action='store_true')
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument(
        "--coarse-ratio",
        default=0.80,
        type=float,
        help="coarse-ratio controls pooling density at each stage")
    parser.add_argument('--dropout', default=0.6, type=float)
    parser.add_argument('--disable_defaults', action='store_true')
    parser.add_argument('--patient', default=400, type=int)
    parser.add_argument('--density', default=0.1, type=float)

    return parser


Tasks = {'diffpool': DiffPoolPredictor}
