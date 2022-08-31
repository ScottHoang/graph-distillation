# pylint: disable=E1101
# pylint: disable=L161
# pylint: disable=W1514
import argparse
import json
import os
import os.path as osp
import sys
from math import ceil
from math import floor
from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torchmetrics as metrics
import tqdm
from easydict import EasyDict as edict
from torch import Tensor
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj

from cold_brew.GNN_model.GNN_normalizations import TeacherGNN
from cold_brew.utils import load_model as load_teacher_model
from lib.data.dataloader import load_data
from lib.pooling_tasks.diffpool import DiffPool
from lib.utils import update_args_dataset


class MLPClassifier(nn.Module):
    """Docstring for  MLPClassifier. """

    def __init__(self, args: Union[Dict, argparse.Namespace]):
        """TODO: to be defined. """
        nn.Module.__init__(self)
        args = vars(args) if isinstance(args, argparse.Namespace) else args
        for k, v in args.items():
            setattr(self, k, v)

        self.k_nodes = ceil(self.num_nodes * self.density)
        self.embeddings = nn.Parameter(
            torch.randn(self.k_nodes, self.L_dim + self.num_classes), True)
        self.lin1 = nn.Linear(self.num_features, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.L_dim + self.num_classes)

        nn.init.orthogonal_(self.embeddings)

    def forward(self, x: Tensor, adj: Tensor, ses: List[Tensor]) -> Tensor:
        x = F.relu(
            self.lin1(F.dropout(x, self.dropout, training=self.training)))
        x = F.relu(self.lin2(F.dropout(x, self.dropout, self.training)))

        if ses is not None:
            embs = self.unpool(ses, self.embeddings)
            x = x + embs

        return x[:, 0:self.num_classes], x[:, self.num_classes::]

    def unpool(self, ses: List[Tensor], weight: Tensor) -> Tensor:
        """
        assuming ses are a list of assignment matrixes going from fine to coarse in order
        obtained from diffpool or another pooling method
        """
        ses = ses[::-1]
        for s in ses:
            s = F.softmax(s, dim=1)
            weight = s @ weight
        return weight


class MLPPredictor(object):

    def __init__(self, args: argparse.Namespace, which_split: int):
        super().__init__()

        self.args = vars(args) if isinstance(args,
                                             argparse.Namespace) else args
        self.args = edict(args)
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(args['device']),
        ])
        self.which_split = which_split
        self.data = load_data(args['dataset'], which_split, transform)
        args = update_args_dataset(self.data, args)

        for k, v in args.items():
            setattr(self, k, v)

        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(self.device),
        ])
        self.data = load_data(self.dataset, which_split, transform)
        args = update_args_dataset(self.data, args)

        self.which_split = which_split

        self.teacher = self.get_teacher_model().to(self.device)
        self.pool = self.get_pool_model().to(self.device)
        #
        self.model = MLPClassifier(self.args).to(self.device)

        with open(osp.join(self.configdir, 'config.txt'), 'w') as handle:
            json.dump(args.__dict__, handle, indent=2)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=5e-4)

        self.loss_fn = F.nll_loss
        self.accuracy = metrics.Accuracy(num_classes=self.num_classes).to(
            self.device)
        self.f1 = metrics.F1Score(num_classes=self.num_classes).to(self.device)
        self.aucroc = metrics.AUROC(num_classes=self.num_classes).to(
            self.device)

        self.teacher.eval()
        self.pool.eval()

    def get_pool_model(self):
        """
        loading the pooling model. if there is a mismatch of density ratio we will correct it
        """
        with open(osp.join(self.pooling_network, 'config.txt'), 'r') as handle:
            config = edict(json.load(handle))
        config.num_features = self.num_features
        config.num_classes = self.num_classes
        config.num_nodes = self.num_nodes
        pooling_network = DiffPool(config)

        path2file = self.pooling_network.replace('configs', 'results')

        model_file = torch.load(osp.join(path2file, str(self.which_split),
                                         'model.pth.tar'),
                                map_location='cpu')
        pooling_network.load_state_dict(model_file)

        if config.density != self.args.density:
            print(f"Resolving pooling density mismatch between \
                {config.density} vs {self.args.density}")
            self.args.density = config.density
            self.density = config.density

        return pooling_network

    def get_teacher_model(self):
        """
        loading teacher model. If there is a mismatch in embeddings we will correct it
        """
        with open(osp.join(self.teacher_network, 'config.txt'), 'r') as handle:
            config = edict(json.load(handle))
        config.dim_commonEmb = self.num_classes
        config.device = self.device
        teacher = TeacherGNN(config)
        teacher.load_state_dict(
            torch.load(osp.join(self.teacher_network, 'teacherGNN.pth'),
                       map_location='cpu'))
        if config.hidden_dim != self.args.L_dim:
            print(
                f"Resolving mismatch teacher/student embeddings {config.hidden_dim} vs {self.args.L_dim}"
            )
            self.args.L_dim = config.hidden_dim
            self.L_dim = config.hidden_dim
        return teacher

    @torch.no_grad()
    def test(self):
        # f1 = metrics.F1Score(num_classes=self.num_classes)

        self.model.eval()
        data = self.data

        assignments = self.pool.pool(data.x, data.edge_index)[-1]
        scores, _ = self.model.forward(data.x, data.edge_index, assignments)
        logits = F.log_softmax(scores, 1)

        results = {}
        masks = ['train', 'val', 'test']
        for m in masks:
            m_vector = getattr(data, f"{m}_mask")
            results[f"{m}_acc"] = self.accuracy.forward(
                logits[m_vector], data.y[m_vector]).item()
            results[f'{m}_f1'] = self.f1.forward(logits[m_vector],
                                                 data.y[m_vector]).item()
            results[f'{m}_aucroc'] = self.aucroc.forward(
                logits[m_vector], data.y[m_vector]).item()

        val_loss = self.loss_fn(logits[self.data.val_mask],
                                self.data.y[self.data.val_mask])
        return results, val_loss.item()

    def train(self):
        self.model.train()
        data = self.data
        with torch.no_grad():
            teacher_output = self.teacher.forward(data.x, data.edge_index)
            teacher_embs = self.teacher.get_embs()
            assignments = self.pool.pool(data.x, data.edge_index)[-1]

        scores, embeddings = self.model.forward(data.x, data.edge_index,
                                                assignments)

        logits = F.log_softmax(scores, 1)
        loss = F.kl_div(logits, teacher_output) + self.loss_fn(
            logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss2 = F.mse_loss(embeddings, teacher_embs)

        overall_loss = loss + loss2
        self.optimizer.zero_grad()
        overall_loss.backward()
        self.optimizer.step()

        return overall_loss.item(), loss.item(), loss2.item()

    def fit(self, writer=None):
        pbar = tqdm.tqdm(range(1, self.epochs + 1),
                         total=self.epochs,
                         desc=f"{self.task.lower()}")
        best_stats = {}
        patient = self.patient
        best_val_loss = float('inf')
        for epoch in pbar:
            if patient <= 0:
                break
            overall_loss, score_loss, embs_loss = self.train()
            results, val_loss = self.test()
            if writer is not None:
                writer.add_scalar(
                    f'Pool{self.task.lower()}-{self.which_split}/overall_loss',
                    overall_loss, epoch)  # new line
                writer.add_scalar(
                    f'Pool{self.task.lower()}-{self.which_split}/loss_scores',
                    score_loss, epoch)  # new line
                writer.add_scalar(
                    f'Pool{self.task.lower()}-{self.which_split}/loss_embs',
                    embs_loss, epoch)  # new line
                for k, v in results.items():
                    writer.add_scalar(
                        f'Pool{self.task.lower()}-{self.which_split}/{k}', v,
                        epoch)  # new line
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_stats = results
                patient = self.patient
                # self.save(True)
            else:
                patient -= 1
            pbar.set_description(
                f"{self.task.lower()} loss:{overall_loss:.4f} ")
        return best_stats

    def save(self, is_best=False):
        """TODO: Docstring for save.
        :returns: TODO

        """
        print(f"saving assignment matrix")
        os.makedirs(osp.join(self.resultdir, str(self.which_split)),
                    exist_ok=True)
        torch.save(
            self.model.state_dict(),
            osp.join(self.resultdir, str(self.which_split), f"model.pth.tar"))
        if is_best:
            torch.save(
                self.model.state_dict(),
                osp.join(self.resultdir, str(self.which_split),
                         f"best_model.pth.tar"))
