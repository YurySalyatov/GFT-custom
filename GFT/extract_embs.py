import os
import torch
import numpy as np
from model.encoder import Encoder
from model.vq import VectorQuantize
from utils.others import load_params
from torch_geometric.data import Data

def load_pretrained_model(params, checkpoint_dir, epoch):
    """Загружает предобученный encoder и vq."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    encoder = load_params(encoder, os.path.join(checkpoint_dir, f'encoder_{epoch}.pt'))
    vq = load_params(vq, os.path.join(checkpoint_dir, f'vq_{epoch}.pt'))

    encoder.to(device).eval()
    vq.to(device).eval()

    return encoder, vq, device


def load_graph_with_mapping(data_path):
    """
    Загружает граф и возвращает data и node_mapping.
    """
    data_obj = torch.load(data_path)
    if isinstance(data_obj, tuple) and len(data_obj) == 2:
        data, _ = data_obj
    else:
        data = data_obj

    # Извлекаем node_mapping, если оно есть
    node_mapping = getattr(data, 'node_mapping', None)
    if node_mapping is None:
        raise ValueError("node_mapping не найден в данных.")

    # Добавляем x и xe если нужно
    if not hasattr(data, 'x'):
        data.x = torch.arange(data.node_text_feat.size(0), dtype=torch.long)
    if not hasattr(data, 'xe'):
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.xe = data.edge_attr.argmax(dim=1)
        else:
            data.xe = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    return data, node_mapping


def extract_embeddings_by_type(encoder, vq, data, node_mapping, device, return_quantized=True):
    """
    Извлекает эмбеддинги для узлов, указанных в node_mapping, сортирует по hetero_index.

    Аргументы:
        encoder, vq: загруженные модели
        data: объект Data
        node_mapping: словарь {(type, hetero_idx): homo_idx}
        device: torch device
        return_quantized: если True, возвращает quantize (дискретные), иначе z (непрерывные)

    Возвращает:
        dict: {type: {'hetero_indices': [int], 'embeddings': np.ndarray (N_type x dim)}}
    """
    with torch.no_grad():
        x = data.node_text_feat.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_text_feat[data.xe].to(device)

        z = encoder(x, edge_index, edge_attr)          # [N, hidden_dim]
        quantize, indices, commit_loss, codes = vq(z)  # quantize: [N, hidden_dim]

        # Выбираем, что использовать
        if return_quantized:
            embeddings_all = quantize.cpu().numpy()
        else:
            embeddings_all = z.cpu().numpy()

        # Группируем по типам
        type_to_pairs = {}
        for (ntype, hetero_idx), homo_idx in node_mapping.items():
            if ntype not in type_to_pairs:
                type_to_pairs[ntype] = []
            type_to_pairs[ntype].append((hetero_idx, homo_idx))
        
        result = {}
        for ntype, pairs in type_to_pairs.items():
            # Сортируем по hetero_idx
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            hetero_indices = [p[0] for p in pairs_sorted]
            homo_indices = [p[1] for p in pairs_sorted]
            # Извлекаем эмбеддинги
            emb_list = [embeddings_all[h] for h in homo_indices]
            emb_array = np.stack(emb_list, axis=0)  # (num_nodes, dim)

            result[ntype] = {
                'hetero_indices': hetero_indices,
                'embeddings': emb_array,
            }

        return result


def main():
    print('start extract')
    # Параметры модели (должны совпадать с pretrain)
    params = {
        'input_dim': 768,
        'hidden_dim': 768,
        'num_layers': 2,
        'activation': 'relu',
        'backbone': 'sage',
        'normalize': 'batch',
        'dropout': 0.15,
        'codebook_size': 128,
        'code_dim': 768,
        'codebook_head': 4,
    }
    import os
    checkpoint_dir = f'{os.getcwd()}/ckpts/pretrain_model/codebook_size_128_layer_2_pretrain_on_all_seed_42'
    epoch = 18
    graph_path = f'{os.getcwd()}/data/pubmed/processed/geometric_data_processed.pt'
    output_dir = './'

    print("Загрузка модели...")
    encoder, vq, device = load_pretrained_model(params, checkpoint_dir, epoch)

    print("Загрузка графа и node_mapping...")
    data, node_mapping = load_graph_with_mapping(graph_path)
    print(f"Всего узлов в графе: {data.node_text_feat.shape[0]}")
    print(f"Количество записей в node_mapping: {len(node_mapping)}")

    print("Извлечение дискретных эмбеддингов по типам...")
    result = extract_embeddings_by_type(encoder, vq, data, node_mapping, device, return_quantized=True)

    # Сохраняем результат в .npz
    output_data = {}
    for ntype, info in result.items():
        output_data[f'{ntype}_embeddings'] = info['embeddings']
        output_data[f'{ntype}_hetero_indices'] = np.array(info['hetero_indices'])

    np.savez(os.path.join(output_dir, 'embeddings_by_type.npz'), **output_data)

    # Также сохраняем в читаемом формате для удобства (например, pickle)
    import pickle
    with open(os.path.join(output_dir, 'embeddings_by_type.pkl'), 'wb') as f:
        pickle.dump(result, f)

    print(f"Результаты сохранены в {output_dir}")
    for ntype, info in result.items():
        print(f"Тип '{ntype}': {info['embeddings'].shape[0]} узлов, размерность {info['embeddings'].shape[1]}")

main()