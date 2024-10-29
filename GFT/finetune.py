#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset.process_datasets import get_finetune_graph
from model.encoder import Encoder
from model.vq import VectorQuantize
from model.ft_model import TaskModel
from utils.loader import get_loader
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.args import get_args_finetune
from utils.preprocess import pre_node, pre_link, pre_graph
from utils.others import seed_everything, load_params, mask2idx, get_split, get_split_graph

from task.node import ft_node, eval_node, ft_node_batch, eval_node_batch
from task.link import ft_link, eval_link, ft_link_batch, eval_link_batch
from task.graph import ft_graph, eval_graph

import warnings
import wandb

warnings.filterwarnings("ignore")

DATASET2TASK = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
    "WN18RR": "link",
    "FB15K237": "link",
    "chemhiv": "graph",
    "chempcba": "graph",
}


def get_preprocess(params):
    if params['task'] == 'node':
        return pre_node
    elif params['task'] == 'link':
        return pre_link
    elif params['task'] == 'graph':
        return pre_graph
    else:
        raise NotImplementedError('The task is not implemented')


def get_ft(params):
    task = params['task']
    batch_size = params["batch_size"]

    if task == "node":
        return ft_node if batch_size == 0 else ft_node_batch
    elif task == "link":
        return ft_link if batch_size == 0 else ft_link_batch
    elif task == "graph":
        return ft_graph
    else:
        raise ValueError("Invalid Task")


def get_eval(params):
    task = params['task']
    batch_size = params["batch_size"]

    if task == "node":
        return eval_node if batch_size == 0 else eval_node_batch
    elif task == "link":
        return eval_link if batch_size == 0 else eval_link_batch
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Invalid Task")



def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_name = params["finetune_dataset"]
    task = params["task"]
    setting = params["setting"]

    dataset, splits, labels, num_classes, num_tasks = get_finetune_graph(data_name)
    num_classes = num_tasks if task == "graph" else num_classes
    preprocess = get_preprocess(params)
    dataset = preprocess(dataset)
    data = dataset[0]
    data.y = labels

    if isinstance(splits, list):
        pass
    elif isinstance(splits, dict):
        splits = [splits] * params["repeat"]

    finetune = get_ft(params)
    evaluate = get_eval(params)

    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )

    vq = VectorQuantize(
        dim=params["hidden_dim"],
        codebook_size=params["codebook_size"],
        codebook_dim=params["code_dim"],
        heads=params["codebook_head"],
        separate_codebook_per_head=True,
        decay=params["codebook_decay"],
        commitment_weight=params["commit_weight"],
        use_cosine_sim=True,  # Cosine Codebook Works, Euclidean Codebook Collapses
        orthogonal_reg_weight=params["ortho_reg_weight"],
        orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
        orthogonal_reg_active_codes_only=False,
        kmeans_init=True,
        ema_update=False,
    )

    # Load Pretrained Model
    if params["pretrain_dataset"] != 'na':
        pretrain_task = params['pretrain_task']

        if pretrain_task == 'all':
            path = osp.join(params['pt_model_path'], "codebook_size_{}_layer_{}_pretrain_on_{}".format(
                params["codebook_size"], params["num_layers"], params["pretrain_dataset"]
            ))
        elif pretrain_task in ['feat_only', 'topo_only', 'sem_only', 'no_codebook', 'no_aug', 'no_ortho_reg']:
            path = osp.join(params['ablation_model_path'], "{}".format(pretrain_task))
        else:
            raise ValueError("Invalid Pretrain Task")

        encoder = load_params(encoder, osp.join(path, f'encoder_{params["pretrain_model_epoch"]}.pt'))
        vq = load_params(vq, osp.join(path, f'vq_{params["pretrain_model_epoch"]}.pt'))

        print("Loader the pretrained encoder and vq model from {}".format(path))

    train_loader = None
    val_loader = None
    test_loader = None
    subgraph_loader = None

    if params["batch_size"] == 0:
        data = data.to(device)
        labels = labels.to(device)

    logger = Logger()

    for idx, split in enumerate(splits):
        seed_everything(idx)

        if setting == "standard":
            split = split
        elif setting in ["few_shot", "zero_shot", "in_context"]:
            if task in ["node", "link"]:
                split = get_split(split, labels, params)
            elif task == "graph":
                split = get_split_graph(split, labels, params)
        else:
            raise ValueError("Invalid Setting")

        task_model = TaskModel(
            encoder=deepcopy(encoder),
            vq=deepcopy(vq),
            num_classes=num_classes,
            params=params,
        ).to(device)

        opt_params = task_model.parameters()
        task_opt = AdamW(opt_params, lr=params["finetune_lr"])
        stopper = EarlyStopping(patience=params["early_stop"])

        if params["batch_size"] != 0 and task in ["node", "link"]:
            train_loader, subgraph_loader = get_loader(data, split, labels, params)
        elif params["batch_size"] != 0 and task == "graph":
            train_loader, val_loader, test_loader = get_loader(dataset, split, labels, params)

        for epoch in range(params["finetune_epochs"]):
            loss = finetune(
                model=task_model,
                dataset=data if task in ["node", "link"] else dataset,
                loader=train_loader,
                split=split,
                labels=labels,
                num_classes=num_classes,
                params=params,
                optimizer=task_opt,
                num_neighbors=[30] * params["num_layers"],
                setting=setting,
                query_node_code_first=params["query_node_code_first"],
            )

            result = evaluate(
                model=task_model,
                dataset=data if task in ["node", "link"] else dataset,
                loader=subgraph_loader,
                split=split,
                labels=labels,
                num_classes=num_classes,
                params=params,
                num_neighbors=[-1] * params["num_layers"],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                setting=setting,
                query_node_code_first=params["query_node_code_first"],
            )

            is_stop = stopper(result)
            logger.log(idx, epoch, loss, result)
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

            wandb.log(
                {
                    "train/proto_loss": loss['proto_loss'],
                    "train/proto_reg": loss['proto_reg'],
                    "train/act_loss": loss['act_loss'],
                    "train/loss": loss['loss'],
                    "train/train_acc": result['train'],
                    "train/val_acc": result['val'],
                    "train/test_acc": result['test'],
                }
            )

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best/train": single_best["train"],
            "best/val": single_best["val"],
            "best/test": single_best["test"],
        })

    best = logger.get_best()

    wandb.log({
        "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final/train_mean": best['train']['mean'],
        "final/val_mean": best['val']['mean'],
        "final/test_mean": best['test']['mean'],
        "final/train_std": best['train']['std'],
        "final/val_std": best['val']['std'],
        "final/test_std": best['test']['std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})

    wandb.finish()


if __name__ == "__main__":
    params = get_args_finetune()

    params['data_path'] = ''
    params['pt_model_path'] = "ckpts/pretrain_model/"
    params['ablation_model_path'] = "ckpts/ablation_model/"
    params['ft_model_path'] = "ckpts/finetune_model/"

    dataset = params["finetune_dataset"]
    task = DATASET2TASK[dataset]
    params['task'] = task

    if params["use_params"]:
        with open("config/finetune.yaml", "r") as f:
            default_params = yaml.safe_load(f)
            params.update(default_params[task][dataset])

    tags = [params['setting']]

    if params["setting"] in ["few_shot"]:
        if params['finetune_dataset'] in ['FB15K237']:
            params['batch_size'] = 0
        if task == 'graph':
            params['n_way'] = 2
            params['num_instances_per_class'] = params['n_train']

    if params['ablation_trade_off'] != -1:
        params['trade_off'] = params['ablation_trade_off']

    if params['ablation_lambda_proto'] != -1:
        params['lambda_proto'] = params['ablation_lambda_proto']

    if params['ablation_lambda_act'] != -1:
        params['lambda_act'] = params['ablation_lambda_act']

    wandb.init(
        project="GFT-Finetune",
        name="{} - Pretrain Epoch {}".format(str.upper(params["finetune_dataset"]), params["pretrain_model_epoch"]),
        config=params,
        mode="disabled" if params["debug"] else "online",  # sweep only works in online mode
        tags=tags,
    )
    params = dict(wandb.config)
    print(params)

    run(params)
