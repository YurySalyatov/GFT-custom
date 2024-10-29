import datetime
from lib2to3.pytree import BasePattern
import numpy as np
import os
import os.path as osp
import random
import logging
from pathlib import Path
import json

import time
import string
import warnings
from contextlib import contextmanager
from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS, TUDataset, WikipediaNetwork, Actor, PPI, \
    Reddit, Flickr, Twitch, Airports

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import transforms as T
from torch_geometric.utils import (degree, remove_self_loops, add_self_loops, to_undirected, k_hop_subgraph, coalesce,
                                   to_edge_index, to_torch_coo_tensor, is_undirected, to_dense_adj)
from torch_geometric.data import Data

import typing

from model.encoder import Encoder
from model.vq import VectorQuantize

EPS = 1e-6


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# Work for acm, dblp, and arxiv-time
def source_sampling(dataset, train_ratio=0.6, valid_ratio=0.2):
    return sampling(dataset, train_ratio, valid_ratio)


def target_sampling(dataset, train_ratio=0.1, valid_ratio=0.1):
    return sampling(dataset, train_ratio, valid_ratio)


def DA_sampling(dataset, train_ratio=0.0, valid_ratio=0.2):
    return sampling(dataset, train_ratio, valid_ratio)


def sampling(dataset, train_ratio=0.1, valid_ratio=0.1):
    tgt_idx = dataset.tgt_mask

    y = dataset.y[tgt_idx].cpu().numpy()
    num_classes = np.unique(y)
    class_index = []
    for i in num_classes:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    train_mask = np.array([])
    valid_mask = np.array([])
    test_mask = np.array([])

    for idx in class_index:
        np.random.shuffle(idx)
        if train_ratio != 0.0:
            train_split = int(len(idx) * train_ratio)
            valid_split = int(len(idx) * (train_ratio + valid_ratio))

            train_mask = np.concatenate((train_mask, idx[:train_split]))
            valid_mask = np.concatenate((valid_mask, idx[train_split:valid_split]))
            test_mask = np.concatenate((test_mask, idx[valid_split:]))
        else:
            valid_split = int(len(idx) * valid_ratio)
            train_mask = None
            valid_mask = np.concatenate((valid_mask, idx[:valid_split]))
            test_mask = np.concatenate((test_mask, idx[valid_split:]))

    train_mask = train_mask.astype(int) if train_mask is not None else None
    valid_mask = valid_mask.astype(int)
    test_mask = test_mask.astype(int)

    return {'train': train_mask, 'valid': valid_mask, 'test': test_mask}


def degree_bucketing(num_nodes, in_degree, max_degree=32):
    features = torch.zeros([num_nodes, max_degree])
    for i in range(num_nodes):
        try:
            features[i][min(in_degree[i], max_degree - 1)] = 1
        except:
            features[i][0] = 1
    return features


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_MB(byte):
    return byte / 1024.0 / 1024.0


def combine_dicts(dicts, decimals=2):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    final_result = {}
    for key, value in result.items():
        if isinstance(value[0], list):
            final_result[key + '_mean'] = np.round(np.mean(value, axis=0), decimals)
            final_result[key + '_std'] = np.round(np.std(value, axis=0), decimals)
        else:
            final_result[key + '_mean'] = np.round(np.mean(value), decimals)
            final_result[key + '_std'] = np.round(np.std(value), decimals)

    return final_result


def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_mask(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.1):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def extract_graphs(data, mask, k=2):
    graphs = list()
    for i in torch.where(mask == True)[0]:
        subgraph = k_hop_subgraph(i.item(), num_hops=k, edge_index=data.edge_index, relabel_nodes=True)
        graphs.append(Data(x=data.x[subgraph[0]], edge_index=subgraph[1], y=data.y[i]))
    return graphs


def get_pooling_graph(data, params):
    sampling = params['sampling']
    if sampling == 'k_hop':
        return get_k_hop_graphs(data, params['hops'], use_self_loop=params['use_self_loop'])
    elif sampling == 'rw':
        if params['rw_mode'] == 'standard':
            return rw_edge_index(data, params['hops'], params['repeat'], params['symm'], p=1, q=1,
                                 use_self_loop=params['use_self_loop'])
        elif params['rw_mode'] == 'local':
            return rw_edge_index(data, params['hops'], params['repeat'], params['symm'], p=5, q=1,
                                 use_self_loop=params['use_self_loop'])
        elif params['rw_mode'] == 'global':
            return rw_edge_index(data, params['hops'], params['repeat'], params['symm'], p=1, q=2,
                                 use_self_loop=params['use_self_loop'])
    else:
        raise NotImplementedError('The sampling method is not supported!')


def get_k_hop_graphs(data, k, use_self_loop=True):
    edge_index, edge_attr = data.edge_index, data.edge_attr
    N = data.x.shape[0]

    if k == 0:
        # return adjacent matrix only with self-loop
        edge_index = [list(range(N)), list(range(N))]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # edge_attr = torch.ones(edge_index.shape[1], device=edge_index.device)
        return edge_index, None

    adj = to_torch_coo_tensor(edge_index, size=(N, N))
    adj_base = adj.clone()

    for _ in range(k - 1):
        new_edge_index, _ = to_edge_index(adj_base @ adj)
        new_edge_index, _ = remove_self_loops(new_edge_index)

        edge_index = torch.cat([edge_index, new_edge_index], dim=1)

    if edge_attr is None:
        # edge_attr = torch.ones(edge_index.shape[1], device=edge_index.device)
        pass

    if not use_self_loop:
        edge_index, _ = remove_self_loops(edge_index)

    return coalesce(edge_index, edge_attr, N)


def rw_edge_index(data, walk_length=10, repeat=1, symm=False, p=1, q=1, use_self_loop=True):
    from torch_cluster import random_walk
    device = data.x.device
    edge_attr = None

    start = torch.arange(data.num_nodes, device=device)
    start = start.view(-1, 1).repeat(1, repeat).view(-1)

    walk = random_walk(data.edge_index[0], data.edge_index[1], start, walk_length, num_nodes=data.num_nodes, p=p, q=q)

    n_mask = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.bool, device=device)

    start = start.view(-1, 1).repeat(1, (walk_length + 1)).view(-1)
    n_mask[start, walk.view(-1)] = True

    if symm:
        n_mask = n_mask | n_mask.t()

    edge_index = n_mask.nonzero().t()

    if not use_self_loop:
        edge_index, _ = remove_self_loops(edge_index)

    return edge_index, edge_attr


def get_k_shot_idx(data, k=5):
    train_classes = data.y[data.train_mask]
    num_samples_per_class = torch.bincount(train_classes)[0].item()
    train_classes, indices = torch.sort(train_classes)

    indices = indices.reshape(-1, num_samples_per_class)
    perm_indices = [index[torch.randperm(indices.shape[1])] for index in indices]
    perm_indices = torch.stack(perm_indices, dim=0)

    perm_indices = perm_indices[:, :k]

    return perm_indices


def random_string(k=16):
    random_string = str.join('', random.choices(string.ascii_letters + string.digits, k=k))
    return random_string


class CMD():
    def mmatch(self, x1, x2, n_moments=5):
        mx1 = x1.mean(0)
        mx2 = x2.mean(0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        # scms = [dm]
        scms = dm
        for i in range(n_moments - 1):
            # moment diff of centralized samples
            # scms.append(self.moment_diff(sx1, sx2, i+2))
            scms += self.moment_diff(sx1, sx2, i + 2)
        # return sum(scms)
        return scms

    def moment_diff(self, sx1, sx2, k):
        """
        difference between moments
        """
        ss1 = sx1.pow(k).mean(0)
        ss2 = sx2.pow(k).mean(0)
        # ss1 = sx1.mean(0)
        # ss2 = sx2.mean(0)
        return self.matchnorm(ss1, ss2)

    def matchnorm(self, x1, x2):
        return (x1 - x2).norm(p=2)

    #    return T.abs_(x1 - x2).sum()# maximum
    #    return 1-T.minimum(x1,x2).sum()/T.maximum(x1,x2).sum()# ruzicka
    #    return kl_divergence(x1,x2)# KL-divergence

    def mmd(self, x1, x2, beta=1.0):
        x1x1 = self.gaussian_kernel(x1, x1, beta)
        x1x2 = self.gaussian_kernel(x1, x2, beta)
        x2x2 = self.gaussian_kernel(x2, x2, beta)
        diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
        return diff

    def gaussian_kernel(self, x1, x2, beta=1.0):
        # r = x1.dimshuffle(0,'x',1)
        r = x1.view(x1.shape[0], 1, x1.shape[1])
        return torch.exp(-beta * torch.square(r - x2).sum(axis=-1))

    def pairwise_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # dist = torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[0]
        y_points = y.shape[0]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def flip_edges(data, p=0.2):
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]

    if is_undirected(data.edge_index):
        num_flip_edges = int(num_edges * p / 2)
    else:
        num_flip_edges = int(num_edges * p)

    adj = to_dense_adj(data.edge_index)[0]

    flipped_edges = torch.randint(0, num_nodes, size=(num_flip_edges, 2))

    for n1, n2 in flipped_edges:
        adj[n1, n2] = 1 - adj[n1, n2]
        adj[n2, n1] = 1 - adj[n2, n1]

    edge_index = adj.to_sparse().coalesce().indices()
    data.edge_index = edge_index
    data.edge_attr = None
    return data


def get_device(params, optimized_params=None):
    if optimized_params is None or len(optimized_params) == 0:
        device = torch.device(f"cuda:{params['device']}")
    else:
        device = torch.device(f"cuda")
    return device


def get_scheduler(optimizer, use_scheduler=True, epochs=1000):
    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    return scheduler


def get_device_from_model(model):
    return next(model.parameters()).device


def active_code(encoder, vq, data):
    z = encoder(data.x, data.edge_index, data.edge_attr)
    _, indices, _, _ = vq(z)
    codebook_size = vq.codebook_size
    codebook_head = vq.heads
    return indices.unique(), indices.unique().numel() / (codebook_size * codebook_head)


def load_params(model, path):
    if isinstance(model, Encoder):
        model.load_state_dict(torch.load(path))
    elif isinstance(model, VectorQuantize):
        z = torch.randn(100, model.dim)
        model(z)
        model.load_state_dict(torch.load(path))
    return model


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def visualize(embedding, label=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_embedded = TSNE(n_components=2).fit_transform(embedding)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label, cmap='tab10')
    plt.show()


def sample_proto_instances(labels, split, num_instances_per_class=10):
    y = labels.cpu().numpy()
    target_y = y[split]
    classes = np.unique(target_y)

    class_index = []
    for i in classes:
        c_i = np.where(y == i)[0]
        c_i = np.intersect1d(c_i, split)
        class_index.append(c_i)

    proto_idx = np.array([])

    for idx in class_index:
        np.random.shuffle(idx)
        proto_idx = np.concatenate((proto_idx, idx[:num_instances_per_class]))

    return proto_idx.astype(int)


def sample_proto_instances_for_graph(labels, split, num_instances_per_class=10):
    y = labels
    ndim = y.ndim
    if ndim == 1:
        y = y.reshape(-1, 1)

    # Map class and instance indices

    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    target_y = y[split]
    task_list = target_y.shape[1]

    # class_index_pos = {}
    # class_index_neg = {}
    task_index_pos, task_index_neg = [], []
    for i in range(task_list):
        c_i = np.where(y[:, i] == 1)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_pos.append(c_i)

        c_i = np.where(y[:, i] == 0)[0]
        c_i = np.intersect1d(c_i, split)
        task_index_neg.append(c_i)

    assert len(task_index_pos) == len(task_index_neg)

    # Randomly select instances for each task

    proto_idx, proto_labels = {}, {}
    for task, (idx_pos, idx_neg) in enumerate(zip(task_index_pos, task_index_neg)):
        tmp_proto_idx, tmp_labels = np.array([]), np.array([])

        # Randomly select instance for the task

        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)
        idx_pos = idx_pos[:num_instances_per_class]
        idx_neg = idx_neg[:num_instances_per_class]

        # Store the randomly selected instances

        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_pos))
        tmp_labels = np.concatenate((tmp_labels, np.ones(len(idx_pos))))
        tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_neg))
        tmp_labels = np.concatenate((tmp_labels, np.zeros(len(idx_neg))))

        proto_idx[task] = tmp_proto_idx.astype(int)
        proto_labels[task] = tmp_labels.astype(int)

    return proto_idx, proto_labels


def mask2idx(mask):
    return torch.where(mask == True)[0]


def idx2mask(idx, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


###################### Few-shot Zero-shot In-context ######################

# Get shared labels between train, valid, and test
def get_shared_labels(train_labels, valid_labels, test_labels, n_shot, n_query):
    base_shared_labels = np.intersect1d(
        np.unique(train_labels), np.unique(valid_labels)
    )
    base_shared_labels = np.intersect1d(base_shared_labels, np.unique(test_labels))

    shared_labels = []
    for label in base_shared_labels:
        if (
                np.sum(train_labels == label) >= n_shot
                and np.sum(valid_labels == label) >= n_query
                and np.sum(test_labels == label) >= n_query
        ):
            shared_labels.append(label)
    return shared_labels


# Randomly select n_way shared labels
def get_random_shared_labels(
        train_labels, valid_labels, test_labels, n_way, n_shot, n_query
):
    shared_labels = get_shared_labels(
        train_labels, valid_labels, test_labels, n_shot, n_query
    )
    n_way = min(n_way, len(shared_labels))
    ways = np.random.choice(shared_labels, n_way, replace=False)
    return ways


# For few-shot setting, select n_train samples for each class. The validation and test sets different.
# For zero-shot setting, does not select n_train samples for each class. The validation and test sets are the same.
# For in-context setting, does not select n_train samples for each class. The validation and test sets are the same.
def get_split(split, labels, params):
    labels = labels.cpu().numpy()
    setting = params["setting"]
    n_task, n_shot, n_way, n_query, n_train = (
        params["n_task"],
        params["n_shot"],
        params["n_way"],
        params["n_query"],
        params["n_train"],
    )
    num_samples = len(labels)
    idx_is_mask = split["train"].dtype == torch.bool

    train_idx, valid_idx, test_idx = (
        mask2idx(split["train"]) if idx_is_mask else split["train"],
        mask2idx(split["valid"]) if idx_is_mask else split["valid"],
        mask2idx(split["test"]) if idx_is_mask else split["test"],
    )
    train_labels, valid_labels, test_labels = (
        labels[train_idx],
        labels[valid_idx],
        labels[test_idx],
    )

    # Select finetune samples for "FEW-SHOT"
    final_train_idx = []
    if setting in ["few_shot"]:
        ways = get_shared_labels(
            train_labels, valid_labels, test_labels, n_train, n_query
        )
        way_indices = [np.where(labels == way)[0] for way in ways]
        for w_idx in way_indices:
            w_idx_train = np.intersect1d(w_idx, train_idx)
            final_train_idx.extend(
                np.random.choice(w_idx_train, n_train, replace=False)
            )
    elif setting in ["zero_shot", "in_context"]:
        print("Does not need to select samples for fine-tuning.")
    final_train_idx = idx2mask(final_train_idx, num_samples)

    # Select test set for "FEW-SHOT", "ZERO-SHOT", and "IN-CONTEXT"
    test_support_idx, test_query_idx = [], []
    for task in range(n_task):
        ways = get_random_shared_labels(
            train_labels, valid_labels, test_labels, n_way, n_shot, n_query
        )
        way_indices = [np.where(labels == way)[0] for way in ways]

        tmp_support_idx, tmp_query_idx = [], []
        for w_idx in way_indices:
            w_idx_support = np.intersect1d(w_idx, train_idx)
            w_idx_support = np.random.choice(w_idx_support, n_shot, replace=False)
            tmp_support_idx.extend(w_idx_support)

            w_idx_query = np.intersect1d(w_idx, test_idx)
            w_idx_query = np.random.choice(w_idx_query, n_query, replace=False)
            tmp_query_idx.extend(w_idx_query)
        test_support_idx.append(idx2mask(tmp_support_idx, num_samples))
        test_query_idx.append(idx2mask(tmp_query_idx, num_samples))

    if setting in ["few_shot"]:
        # The validation set is separate for few-shot learning
        valid_support_idx, valid_query_idx = [], []
        for task in range(n_task):
            ways = get_random_shared_labels(
                train_labels, valid_labels, test_labels, n_way, n_shot, n_query
            )
            way_indices = [np.where(labels == way)[0] for way in ways]

            tmp_support_idx, tmp_query_idx = [], []
            for w_idx in way_indices:
                w_idx_support = np.intersect1d(w_idx, train_idx)
                w_idx_support = np.random.choice(w_idx_support, n_shot, replace=False)
                tmp_support_idx.extend(w_idx_support)

                w_idx_query = np.intersect1d(w_idx, test_idx)
                w_idx_query = np.random.choice(w_idx_query, n_query, replace=False)
                tmp_query_idx.extend(w_idx_query)
            valid_support_idx.append(idx2mask(tmp_support_idx, num_samples))
            valid_query_idx.append(idx2mask(tmp_query_idx, num_samples))
    elif setting in ["zero_shot", "in_context"]:
        # The validation set is the same as the test set
        valid_support_idx = test_support_idx
        valid_query_idx = test_query_idx

    return {
        "train": final_train_idx,
        "valid": {"support": valid_support_idx, "query": valid_query_idx},
        "test": {"support": test_support_idx, "query": test_query_idx},
    }


def get_split_graph(split, labels, params):
    labels = labels.cpu().numpy()
    setting = params["setting"]
    n_task, n_shot, n_way, n_query, n_train = (
        params["n_task"],
        params["n_shot"],
        params["n_way"],
        params["n_query"],
        params["n_train"],
    )
    num_samples = len(labels)

    train_idx, valid_idx, test_idx = (
        split["train"],
        split["valid"],
        split["test"]
    )
    train_labels, valid_labels, test_labels = (
        labels[train_idx],
        labels[valid_idx],
        labels[test_idx],
    )

    # Select finetune samples for "FEW-SHOT"
    from submission.GFM_Tree_submission_version.finetune.other_utils import sample_proto_instances_for_graph
    select_few_shot_samples = sample_proto_instances_for_graph
    if setting in ["few_shot"]:
        final_train_idx, final_train_labels = select_few_shot_samples(
            labels, train_idx, n_train
        )
    elif setting in ["zero_shot", "in_context"]:
        print("Does not need to select samples for fine-tuning.")

    # Select test set for "FEW-SHOT", "ZERO-SHOT", and "IN-CONTEXT"
    test_support_idx, test_query_idx = [], []
    test_support_label, test_query_label = [], []
    for task in range(n_task):
        tmp_support_idx, tmp_support_label = select_few_shot_samples(
            labels, train_idx, n_shot
        )
        tmp_query_idx, tmp_query_label = select_few_shot_samples(
            labels, test_idx, n_query
        )
        test_support_idx.append(tmp_support_idx)
        test_query_idx.append(tmp_query_idx)
        test_support_label.append(tmp_support_label)
        test_query_label.append(tmp_query_label)

    if setting in ["few_shot"]:
        # The validation set is separate for few-shot learning
        valid_support_idx, valid_query_idx = [], []
        valid_support_label, valid_query_label = [], []
        for task in range(n_task):
            tmp_support_idx, tmp_support_label = select_few_shot_samples(
                labels, train_idx, n_shot
            )
            tmp_query_idx, tmp_query_label = select_few_shot_samples(
                labels, test_idx, n_query
            )
            valid_support_idx.append(tmp_support_idx)
            valid_query_idx.append(tmp_query_idx)
            valid_support_label.append(tmp_support_label)
            valid_query_label.append(tmp_query_label)
    elif setting in ["zero_shot", "in_context"]:
        # The validation set is the same as the test set
        valid_support_idx, valid_query_idx = test_support_idx, test_query_idx
        valid_support_label, valid_query_label = test_support_label, test_query_label

    return {
        "train": [item for sublist in final_train_idx.values() for item in sublist],
        # Here we only update the train_idx
        "valid": {
            "support": {'idx': valid_support_idx, 'label': valid_support_label},
            "query": {'idx': valid_query_idx, 'label': valid_query_label}
        },
        "test": {
            "support": {'idx': test_support_idx, 'label': test_support_label},
            "query": {'idx': test_query_idx, 'label': test_query_label}
        },
    }
