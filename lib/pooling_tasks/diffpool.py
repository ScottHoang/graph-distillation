# pylint: disable=E1101
# pylint: disable=L161
import argparse
import os
import os.path as osp
import sys
from math import ceil
from math import floor
from typing import Dict
from typing import Union

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torchmetrics as metrics
import tqdm
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj

from lib.data.dataloader import load_data
from lib.utils import update_args_dataset


class DiffPoolGCN(torch.nn.Module):
    """
     GCN for Diff pool. Code is modified and obtained from the same source as DiffPool
    class
    """

    def __init__(
        self,
        in_c,
        hidden_c,
        out_c,
        norm=True,
    ):
        """TODO: to be defined. """
        torch.nn.Module.__init__(self)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        prev_input = in_c
        for _ in range(2):
            self.convs.append(GCNConv(prev_input, hidden_c, norm))
            self.bns.append(torch.nn.BatchNorm1d(hidden_c))
            prev_input = hidden_c

        self.convs.append(GCNConv(hidden_c, out_c, norm))
        self.bns.append(torch.nn.BatchNorm1d(out_c))

    def forward(self, x, edge_index, edge_weight=None):
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, edge_index, edge_weight))
        return x


class DiffPool(torch.nn.Module):
    """
    Differential Pooling. Code obtained and modified from
    https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial16/Tutorial16.ipynb
    """

    def __init__(self, args: Union[argparse.Namespace, Dict]):
        """TODO: to be defined. """
        torch.nn.Module.__init__(self)
        args = vars(args) if isinstance(args, argparse.Namespace) else args
        for k, v in args.items():
            setattr(self, k, v)

        self.gnn_pool = torch.nn.ModuleList()
        self.gnn_embed = torch.nn.ModuleList()
        cur_features = self.num_features
        prev_num_nodes = self.num_nodes
        final_nodes = ceil(self.num_nodes * self.density)
        for layer in range(1, self.num_layers + 1):
            if layer < self.num_layers:
                cur_num_nodes = max(ceil(prev_num_nodes * self.coarse_ratio),
                                    final_nodes)
            else:
                # last layer
                cur_num_nodes = final_nodes
            self.gnn_pool.append(
                DiffPoolGCN(cur_features, self.hidden_dim, cur_num_nodes))
            self.gnn_embed.append(
                DiffPoolGCN(cur_features, self.hidden_dim, self.hidden_dim))
            cur_features = self.hidden_dim
            prev_num_nodes = cur_num_nodes

        self.final_nodes = final_nodes
        self.gnnfin_embed = DiffPoolGCN(self.hidden_dim, self.hidden_dim,
                                        self.hidden_dim)
        self.tau = 5.0
        self.tau_step = (5.0 - 0.001) / self.epochs

        self.lin1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.num_classes)

    def pool(self, x, adj):
        ses, ls, es = [], [], []
        edge_weight = None
        for i in range(len(self.gnn_pool)):
            s = self.gnn_pool[i](x, adj, edge_weight)
            x = self.gnn_embed[i](x, adj, edge_weight)
            x, adj, l1, e1 = dense_diff_pool(
                x.unsqueeze(0), to_dense_adj(adj, max_num_nodes=x.size(0)),
                s.unsqueeze(0))
            adj, edge_weight = dense_to_sparse(adj)
            x = x.squeeze()
            ls.append(l1)
            es.append(e1)
            # self.write2buffer(getattr(self, f"s{i}"), s)
            setattr(self, f"_s{i}", s)
            ses.append(s)
        return x, adj, edge_weight, ls, es, ses

    def forward(self, x, adj):
        x, adj, edge_weight, ls, es, ses = self.pool(x, adj)
        x = self.gnnfin_embed(x, adj, edge_weight)
        x = self.unpool(x)
        x = F.relu(self.lin1(F.dropout(x, training=self.training)))
        x = self.lin2(x)

        return x, sum(ls), sum(es)

    # def write2buffer(self, buffer, value):
    # buffer.data.copy_(value.data)

    def unpool(self, weight):
        for i in range(self.num_layers - 1, -1, -1):
            s = getattr(self, f"_s{i}")
            if self.training:
                s = F.gumbel_softmax(s, tau=self.tau, dim=1)
                self.tau = max(0.001, self.tau - self.tau_step)
            else:
                s = F.softmax(s, dim=1)
            weight = s @ weight
        return weight

    def regrow(self, s0, s1, adj):
        s1 = s1.softmax(dim=1)
        s0 = s0.softmax(dim=1)
        adj = s1 @ adj @ s1.T()
        adj = s0 @ adj @ s0.T()
        return adj


class DiffPoolPredictor(object):

    def __init__(self, args: argparse.Namespace, which_split: int):
        super().__init__()

        args = vars(args) if isinstance(args, argparse.Namespace) else args
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

        self.model = DiffPool(args).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=5e-4)

        self.loss_fn = F.nll_loss
        self.accuracy = metrics.Accuracy(num_classes=self.num_classes).to(
            self.device)
        self.f1 = metrics.F1Score(num_classes=self.num_classes).to(self.device)
        self.aucroc = metrics.AUROC(num_classes=self.num_classes).to(
            self.device)

    @torch.no_grad()
    def test(self):
        # f1 = metrics.F1Score(num_classes=self.num_classes)

        self.model.eval()
        data = self.data
        scores, _, _ = self.model.forward(data.x, data.edge_index)
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
        scores, l_loss, e_loss = self.model.forward(data.x, data.edge_index)
        logits = F.log_softmax(scores, 1)
        loss = self.loss_fn(logits[self.data.train_mask],
                            self.data.y[self.data.train_mask])

        loss = loss + l_loss + e_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), l_loss.item(), e_loss.item()

    def fit(self, writer=None):
        pbar = tqdm.tqdm(range(1, self.epochs + 1),
                         total=self.epochs,
                         desc=f"{self.task.lower()}")
        best_stats = {}
        lowest_loss = float('inf')
        patient = self.patient
        for epoch in pbar:
            if patient <= 0:
                break
            overall_loss, link_pred_loss, entropy_loss = self.train()
            results, val_loss = self.test()
            if writer is not None:
                writer.add_scalar(
                    f'Pool{self.task.lower()}-{self.which_split}/loss_train',
                    overall_loss, epoch)  # new line
                writer.add_scalar(
                    f'Pool{self.task.lower()}-{self.which_split}/link_pred_loss',
                    link_pred_loss, epoch)  # new line
                writer.add_scalar(
                    f'Pool{self.task.lower()}-{self.which_split}/entropy_loss',
                    entropy_loss, epoch)  # new line
                writer.add_scalar(
                    f'Pool{self.task.lower()}-{self.which_split}/train_acc',
                    results['train_acc'], epoch)  # new line
            if lowest_loss > overall_loss:
                lowest_loss = overall_loss
                best_stats['overall_loss'] = overall_loss
                best_stats['link_loss'] = link_pred_loss
                best_stats['entropy_loss'] = entropy_loss
                patient = self.patient
            else:
                patient -= 1
            pbar.set_description(
                f"{self.task.lower()} loss:{overall_loss:.4f} edge-loss:{link_pred_loss:.4f} entropy_loss:{entropy_loss:.4f}"
            )
        return best_stats

    def save(self):
        """TODO: Docstring for save.
        :returns: TODO

        """
        print(f"saving assignment matrix")
        os.makedirs(osp.join(self.resultdir, str(self.which_split)),
                    exist_ok=True)
        torch.save(
            self.model.state_dict(),
            osp.join(self.resultdir, str(self.which_split), f"model.pth.tar"))


if __name__ == "__main__":
    pass
    # """ testing """
    # import sys
    # sys.path.append("/home/dnh754/graph-distillation/")
    # from lib.data.dataloader import load_data
    # parser = argparse.ArgumentParser()
    # data = load_data("Cora", 0)
    # num_classes = data.y.max() + 1
    # num_nodes = data.x.size(0)
    # num_features = data.x.size(1)
    # args = {
    # "num_nodes": num_nodes,
    # "hidden": 64,
    # 'num_classes': num_classes,
    # "num_features": num_features,
    # "k_nodes": 256
    # }
    # diffpool = DiffPool(args)
    # diffpool(data.x, data.edge_index)
