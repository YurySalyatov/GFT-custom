import os
import numpy as np

import torch
from torch_geometric.loader import NeighborLoader

from utils.eval import evaluate, task2metric
from utils.others import get_device_from_model, sample_proto_instances, mask2idx


# This works for all settings, including standard, few_shot, zero_shot, in_context
# For standard and few-shot, the model fine-tunes as usual
# For zero-shot, the model does not fine-tune, i.e., returns 0 for all losses
def ft_node(model, dataset, loader, split, labels, num_classes, optimizer, params, scheduler=None, **kwargs):
    model.train()
    device = get_device_from_model(model)

    # Encode

    x = dataset.node_text_feat
    edge_index = dataset.edge_index
    edge_attr = dataset.edge_text_feat[dataset.xe]
    y = labels.to(device)

    z = model.encode(x, edge_index, edge_attr)

    # Compute Prototypes

    train_mask = split["train"]
    z_train, y_train = z[train_mask], y[train_mask]
    code_train, commit_loss = model.get_codes(z_train, use_orig_codes=True)

    proto_emb = model.get_class_prototypes(code_train, y_train, num_classes).detach()
    query_emb = z_train if model.use_z_in_predict else code_train

    # Compute Losses

    proto_loss = model.compute_proto_loss(query_emb, proto_emb, y_train) * params["lambda_proto"]
    proto_reg = model.compute_proto_reg(proto_emb) * params["lambda_proto_reg"]
    act_loss = model.compute_activation_loss(z_train, y_train) * params["lambda_act"]
    loss = proto_loss + proto_reg + act_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return {
        'proto_loss': proto_loss.item(),
        'proto_reg': proto_reg.item(),
        'act_loss': act_loss.item(),
        'loss': loss.item(),
    }


def ft_node_batch(model, dataset, loader, split, labels, num_classes, optimizer, params, scheduler=None, **kwargs):
    model.train()
    device = get_device_from_model(model)

    setting = params["setting"]

    # Define Prototype Loader.

    # This is a unique step for mini-batch training
    # As we cannot use all instances to compute prototypes

    if setting == "standard":
        # You only need to sample instances for standard setting
        # In few-shot, we could use all instances available
        proto_idx = sample_proto_instances(
            labels,
            mask2idx(split["train"]),
            num_instances_per_class=model.num_instances_per_class,
        )
        proto_loader = NeighborLoader(
            dataset,
            num_neighbors=kwargs["num_neighbors"],
            input_nodes=proto_idx,
            batch_size=512,
            num_workers=8,
        )
    elif setting in ["few_shot"]:
        # In few-shot setting, we use the same train_loader for all tasks
        # As the number of few-shot instances is small enough
        proto_loader = loader

    # Compute Prototypes

    code_list, y_list = [], []
    for batch in proto_loader:
        batch = batch.to(device)
        bs = batch.batch_size

        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat[batch.xe]

        y = batch.y[:bs]
        z = model.encode(x, edge_index, edge_attr)[:bs]

        code, _ = model.get_codes(z, use_orig_codes=True)
        code_list.append(code.detach())
        y_list.append(y)

    code = torch.cat(code_list, dim=0)
    y = torch.cat(y_list, dim=0)
    proto_emb = model.get_class_prototypes(code, y, num_classes)

    # Start Training

    total_proto_loss = 0
    total_proto_reg = 0
    total_act_loss = 0
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        bs = batch.batch_size

        # Encode
        x = batch.node_text_feat
        edge_index = batch.edge_index
        edge_attr = batch.edge_text_feat[batch.xe]

        y = batch.y[:bs]
        z = model.encode(x, edge_index, edge_attr)[:bs]

        # Compute Prototypes

        code, commit_loss = model.get_codes(z, use_orig_codes=True)
        query_emb = z if model.use_z_in_predict else code

        # Compute Losses

        proto_loss = model.compute_proto_loss(query_emb, proto_emb, y) * params["lambda_proto"]
        proto_reg = model.compute_proto_reg(proto_emb) * params["lambda_proto_reg"]
        act_loss = model.compute_activation_loss(z, y) * params["lambda_act"]
        loss = proto_loss + proto_reg + act_loss

        total_proto_loss += proto_loss.item()
        total_proto_reg += proto_reg.item()
        total_act_loss += act_loss.item()
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    return {
        'proto_loss': total_proto_loss / len(loader),
        'proto_reg': total_proto_reg / len(loader),
        'act_loss': total_act_loss / len(loader),
        'loss': total_loss / len(loader),
    }


# This works for all settings, including standard, few_shot, zero_shot, in_context
# For standard, there is one a single task
# For few-shot, zero-shot, and in-context, there are multiple tasks
def eval_node(model, dataset, loader, split, labels, num_classes, params, **kwargs):
    model.eval()
    setting = params["setting"]

    # Encode

    x = dataset.node_text_feat
    edge_index = dataset.edge_index
    edge_attr = dataset.edge_text_feat[dataset.xe]
    y = labels.to(x.device)

    z = model.encode(x, edge_index, edge_attr)

    if setting == "standard":

        # Compute Prototypes

        train_mask = split["train"]
        code, _ = model.get_codes(z, use_orig_codes=True)
        code_train, y_train = code[train_mask], y[train_mask]

        proto_emb = model.get_class_prototypes(
            code_train, y_train, num_classes
        ).detach()
        query_emb = z if model.use_z_in_predict else code  # Use all instances

        # Compute logits

        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
        pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
        pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

        # Evaluate
        train_mask, val_mask, test_mask = (
            split["train"],
            split["valid"],
            split["test"],
        )
        train_value = evaluate(pred, y, train_mask, params)
        val_value = evaluate(pred, y, val_mask, params)
        test_value = evaluate(pred, y, test_mask, params)

        return {
            'train': train_value,
            'val': val_value,
            'test': test_value,
            'metric': task2metric[params['task']]
        }

    elif setting in ["few_shot", "zero_shot", "in_context"]:
        valid_as_test = setting in ["zero_shot", "in_context"]
        use_outer_proto_emb = setting in ["zero_shot"]

        n_task = len(split["valid"]["support"])
        train_values, val_values, test_values = [], [], []

        # Validation: few-shot, zero-shot, and in-context
        # For zero-shot and in-context, the validation is the same as the test
        for i in range(n_task):
            s_mask = split["valid"]["support"][i]
            q_mask = split["valid"]["query"][i]

            # Compute Prototypes

            code, _ = model.get_codes(z, use_orig_codes=True)
            code_support, y_support = code[s_mask], y[s_mask]
            z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]

            if use_outer_proto_emb:
                proto_emb = dataset.class_node_text_feat
                proto_emb, _ = model.get_codes(proto_emb, use_orig_codes=True)
            else:
                proto_emb = model.get_class_prototypes(
                    code_support, y_support, num_classes
                ).detach()

            query_emb = z_query if model.use_z_in_predict else code_query

            # Compute logits

            pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
            pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)
            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            # Evaluate

            value = evaluate(pred, y_query, params=params)
            train_values.append(value)
            val_values.append(value)

            if valid_as_test:
                test_values.append(value)

        # Test: few-shot
        if not valid_as_test:
            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]

                # Compute Prototypes

                code, _ = model.get_codes(z, use_orig_codes=True)
                code_support, y_support = code[s_mask], y[s_mask]
                z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]

                proto_emb = model.get_class_prototypes(
                    code_support, y_support, num_classes
                ).detach()

                query_emb = z_query if model.use_z_in_predict else code_query

                # Compute logits

                pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(
                    dim=-1
                )
                pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)
                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate

                value = evaluate(pred, y_query, params=params)
                test_values.append(value)

        return {
            'train': np.mean(train_values),
            'val': np.mean(val_values),
            'test': np.mean(test_values),
            'metric': task2metric[params['task']]
        }


def eval_node_batch(model, dataset, loader, split, labels, num_classes, params, **kwargs):
    model.eval()
    device = get_device_from_model(model)
    setting = params["setting"]

    # The standard setting and the remaining settings are handled differently
    if setting == "standard":

        # Define Prototype Loader
        # Prototype instance sampling only for standard setting
        proto_idx = sample_proto_instances(
            labels,
            mask2idx(split["train"]),
            num_instances_per_class=model.num_instances_per_class,
        )
        proto_loader = NeighborLoader(
            dataset,
            num_neighbors=kwargs["num_neighbors"],
            input_nodes=proto_idx,
            batch_size=256,
            num_workers=8,
        )

        # Encode Prototypes

        code_list, y_list = [], []
        for batch in proto_loader:
            batch = batch.to(device)
            bs = batch.batch_size

            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat[batch.xe]

            y = batch.y[:bs]
            z = model.encode(x, edge_index, edge_attr)[:bs]

            code, _ = model.get_codes(z, use_orig_codes=True)
            code_list.append(code.detach())
            y_list.append(y)

        code = torch.cat(code_list, dim=0)
        y = torch.cat(y_list, dim=0)
        proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

        # Do Prediction

        pred_list, y_list = [], []
        for batch in loader:
            batch = batch.to(device)
            bs = batch.batch_size

            # Encode

            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat[batch.xe]

            y = batch.y[:bs]
            z = model.encode(x, edge_index, edge_attr)[:bs]

            code, _ = model.get_codes(z, use_orig_codes=True)
            query_emb = z if model.use_z_in_predict else code

            # Compute logits

            pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
            pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            pred_list.append(pred.detach())
            y_list.append(y)

        pred = torch.cat(pred_list, dim=0)
        y = torch.cat(y_list, dim=0)

        train_mask, val_mask, test_mask = (
            split["train"],
            split["valid"],
            split["test"],
        )
        train_value = evaluate(pred, y, train_mask, params)
        val_value = evaluate(pred, y, val_mask, params)
        test_value = evaluate(pred, y, test_mask, params)

        return {
            'train': train_value,
            'val': val_value,
            'test': test_value,
            'metric': task2metric[params['task']]
        }

    elif setting in ["few_shot", "zero_shot", "in_context"]:
        valid_as_test = setting in ["zero_shot", "in_context"]
        use_outer_proto_emb = setting in ["zero_shot"]

        n_task = len(split["valid"]["support"])
        train_values, val_values, test_values = [], [], []

        # Validation: few-shot, zero-shot, and in-context
        for i in range(n_task):
            s_mask = split["valid"]["support"][i]
            q_mask = split["valid"]["query"][i]

            # Define Loaders for Support and Query Sets
            # Prototype loader for support set
            # Query loader for query set
            proto_loader = NeighborLoader(
                dataset,
                num_neighbors=kwargs["num_neighbors"],
                input_nodes=mask2idx(s_mask),
                batch_size=256,
                num_workers=8,
            )
            query_loader = NeighborLoader(
                dataset,
                num_neighbors=kwargs["num_neighbors"],
                input_nodes=mask2idx(q_mask),
                batch_size=256,
                num_workers=8,
            )

            # Construct Prototypes based on Support Set

            code_list, y_list = [], []
            for batch in proto_loader:
                batch = batch.to(device)
                bs = batch.batch_size

                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]

                y = batch.y[:bs]
                z = model.encode(x, edge_index, edge_attr)[:bs]

                code, _ = model.get_codes(z, use_orig_codes=True)
                code_list.append(code.detach())
                y_list.append(y)
            code = torch.cat(code_list, dim=0)
            y = torch.cat(y_list, dim=0)

            if use_outer_proto_emb:
                proto_emb = dataset.class_node_text_feat.to(device)
                proto_emb, _ = model.get_codes(proto_emb, use_orig_codes=True)
            else:
                proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

            # Compute logits

            pred_list, y_list = [], []
            for batch in query_loader:
                batch = batch.to(device)
                bs = batch.batch_size

                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]

                y = batch.y[:bs]
                z = model.encode(x, edge_index, edge_attr)[:bs]
                code, _ = model.get_codes(z, use_orig_codes=True)

                query_emb = z if model.use_z_in_predict else code

                pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
                pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                pred_list.append(pred.detach())
                y_list.append(y)

            pred = torch.cat(pred_list, dim=0)
            y = torch.cat(y_list, dim=0)

            value = evaluate(pred, y, params=params)
            train_values.append(value)
            val_values.append(value)
            if valid_as_test:
                test_values.append(value)

        # Test: only few-shot
        if not valid_as_test:
            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]

                # Define Loaders for Support and Query Sets
                # Prototype loader for support set
                # Query loader for query set
                proto_loader = NeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    input_nodes=mask2idx(s_mask),
                    batch_size=256,
                    num_workers=8,
                )
                query_loader = NeighborLoader(
                    dataset,
                    num_neighbors=kwargs["num_neighbors"],
                    input_nodes=mask2idx(q_mask),
                    batch_size=256,
                    num_workers=8,
                )

                # Construct Prototypes based on Support Set

                code_list, y_list = [], []
                for batch in proto_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]

                    code, _ = model.get_codes(z, use_orig_codes=True)
                    code_list.append(code.detach())
                    y_list.append(y)
                code = torch.cat(code_list, dim=0)
                y = torch.cat(y_list, dim=0)

                if use_outer_proto_emb:
                    proto_emb = dataset.class_node_text_feat.to(device)
                    proto_emb, _ = model.get_codes(proto_emb, use_orig_codes=True)
                else:
                    proto_emb = model.get_class_prototypes(code, y, num_classes).detach()

                # Compute logits

                pred_list, y_list = [], []
                for batch in query_loader:
                    batch = batch.to(device)
                    bs = batch.batch_size

                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]

                    y = batch.y[:bs]
                    z = model.encode(x, edge_index, edge_attr)[:bs]
                    code, _ = model.get_codes(z, use_orig_codes=True)

                    query_emb = z if model.use_z_in_predict else code

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
                    pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
                    pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                    pred_list.append(pred.detach())
                    y_list.append(y)

                pred = torch.cat(pred_list, dim=0)
                y = torch.cat(y_list, dim=0)

                value = evaluate(pred, y, params=params)
                test_values.append(value)

        return {
            'train': np.mean(train_values),
            'val': np.mean(val_values),
            'test': np.mean(test_values),
            'metric': task2metric[params['task']]
        }
