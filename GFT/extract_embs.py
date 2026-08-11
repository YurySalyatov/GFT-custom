import os
import torch
import numpy as np
import argparse
import pickle
from model.encoder import Encoder
from model.vq import VectorQuantize
from utils.others import load_params
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


def load_pretrained_model(params, checkpoint_dir, epoch, device):
    encoder = Encoder(
        input_dim=params['input_dim'],
        hidden_dim=params['hidden_dim'],
        activation=torch.nn.ReLU if params['activation'] == 'relu' else torch.nn.LeakyReLU,
        num_layers=params['num_layers'],
        backbone=params['backbone'],
        normalize=params['normalize'],
        dropout=params['dropout'],
    )
    vq = VectorQuantize(
        dim=params['hidden_dim'],
        codebook_size=params['codebook_size'],
        codebook_dim=params['code_dim'],
        heads=params['codebook_head'],
        separate_codebook_per_head=True,
        decay=0.8,
        commitment_weight=10,
        use_cosine_sim=True,
        orthogonal_reg_weight=1,
        orthogonal_reg_max_codes=32,
        kmeans_init=False,
        ema_update=False,
    )
    encoder = load_params(encoder, os.path.join(checkpoint_dir, f'encoder_{epoch}.pt'), device=device)
    vq = load_params(vq, os.path.join(checkpoint_dir, f'vq_{epoch}.pt'), device=device)
    encoder.to(device).eval()
    vq.to(device).eval()
    return encoder, vq


def load_graph_with_mapping(data_path):
    data_obj = torch.load(data_path, weights_only=False)
    if isinstance(data_obj, tuple) and len(data_obj) == 2:
        data, _ = data_obj
    else:
        data = data_obj
    node_mapping = getattr(data, 'node_mapping', None)
    if node_mapping is None:
        raise ValueError("node_mapping не найден в данных.")
    if not hasattr(data, 'x'):
        data.x = torch.arange(data.node_text_feat.size(0), dtype=torch.long)
    if not hasattr(data, 'xe'):
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.xe = data.edge_attr.argmax(dim=1)
        else:
            data.xe = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    return data, node_mapping


def extract_embeddings_by_type_batched(encoder, vq, data, node_mapping, device,
                                       batch_size=1024, num_layers=2, return_quantized=True):
    """
    Извлекает эмбеддинги для всех узлов, обрабатывая их батчами с использованием NeighborLoader.
    """
    # Все целевые узлы (их homo-индексы)
    target_nodes = torch.tensor(list(node_mapping.values()), dtype=torch.long)

    # Создаём загрузчик, который выдаёт подграфы для батчей целевых узлов
    loader = NeighborLoader(
        data,
        input_nodes=target_nodes,
        num_neighbors=[10] * num_layers,
        batch_size=batch_size,
        shuffle=False,  # сохраняем порядок
    )

    # Для сопоставления homo_idx -> (type, hetero_idx)
    homo_to_type_and_hetero = {homo: (ntype, hetero) for (ntype, hetero), homo in node_mapping.items()}

    # Временные хранилища
    type_to_embeddings = {}
    type_to_indices = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            # Перемещаем данные на устройство
            x = batch.node_text_feat.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_text_feat[batch.xe].to(device)

            # Forward pass
            z = encoder(x, edge_index, edge_attr)
            quantize, _, _, _ = vq(z)

            # Берём эмбеддинги только для seed-узлов (первые batch.batch_size)
            seed_embeds = quantize[:batch.batch_size] if return_quantized else z[:batch.batch_size]

            # batch.input_id содержит исходные индексы seed-узлов
            seed_homo = batch.input_id.cpu().numpy()
            for i, homo in enumerate(seed_homo):
                ntype, hetero = homo_to_type_and_hetero[homo]
                type_to_embeddings.setdefault(ntype, []).append(seed_embeds[i].cpu().numpy())
                type_to_indices.setdefault(ntype, []).append(hetero)

    # Формируем результат, сортируя по hetero_idx
    target_type = 'author'
    result = {}
    for ntype in type_to_embeddings:
        if ntype != target_type:
            continue
        pairs = sorted(zip(type_to_indices[ntype], type_to_embeddings[ntype]), key=lambda x: x[0])
        hetero_indices = [p[0] for p in pairs]
        embeddings = np.stack([p[1] for p in pairs], axis=0)
        result[ntype] = {
            'hetero_indices': hetero_indices,
            'embeddings': embeddings,
        }
    return result


def main():
    import yaml
    import os
    import argparse
    import torch
    import numpy as np
    import pickle
    from your_imports import load_pretrained_model, load_graph_with_mapping, extract_embeddings_by_type_batched  # замените на реальные

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Dir with encoder_*.pt and vq_*.pt')
    parser.add_argument('--epoch', type=int, default=50, help='Epoch number to load')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='Root output directory')
    parser.add_argument('--quantized', action='store_true', default=True, help='Use quantized embeddings')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for extraction')
    # Убираем --num_layers, так как теперь берём из конфига
    # parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers (must match training)')
    args = parser.parse_args()

    # ========== Загрузка параметров из YAML ==========
    config_path = '../config/pretrain.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Извлекаем нужные для модели параметры
    params = {
        'input_dim': config['input_dim'],
        'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'],
        'activation': config['activation'],
        'backbone': config['backbone'],
        'normalize': config['normalize'],
        'dropout': config['dropout'],
        'codebook_size': config['codebook_size'],
        'code_dim': config['code_dim'],
        'codebook_head': config['codebook_head'],
    }
    # Если нужно, можно также взять другие параметры, например, для загрузки,
    # но в load_pretrained_model используются только перечисленные выше.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint_dir}, epoch {args.epoch}")
    print(f"Model parameters: {params}")
    encoder, vq = load_pretrained_model(params, args.checkpoint_dir, args.epoch, device)

    graph_path = f'data/{args.dataset}/processed/geometric_data_processed.pt'
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    print(f"Loading graph {args.dataset}...")
    data, node_mapping = load_graph_with_mapping(graph_path)

    print("Extracting embeddings in batches...")
    result = extract_embeddings_by_type_batched(
        encoder, vq, data, node_mapping, device,
        batch_size=args.batch_size,
        num_layers=params['num_layers'],  # теперь берём из конфига
        return_quantized=args.quantized
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_pkl = os.path.join(args.output_dir, f'{args.dataset}_embeddings.pkl')

    output_data = {}
    for ntype, info in result.items():
        output_data[f'{ntype}_embeddings'] = info['embeddings']
        output_data[f'{ntype}_hetero_indices'] = np.array(info['hetero_indices'])
    with open(out_pkl, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved to {args.output_dir}")
    for ntype, info in result.items():
        print(f"Type '{ntype}': {info['embeddings'].shape[0]} nodes, dim {info['embeddings'].shape[1]}")

if __name__ == '__main__':
    main()