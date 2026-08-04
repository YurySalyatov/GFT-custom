import os
import torch
import numpy as np
import argparse
import pickle
from model.encoder import Encoder
from model.vq import VectorQuantize
from utils.others import load_params

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

def extract_embeddings_by_type(encoder, vq, data, node_mapping, device, return_quantized=True):
    with torch.no_grad():
        x = data.node_text_feat.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_text_feat[data.xe].to(device)
        z = encoder(x, edge_index, edge_attr)
        quantize, indices, commit_loss, codes = vq(z)
        embeddings_all = quantize.cpu().numpy() if return_quantized else z.cpu().numpy()
        type_to_pairs = {}
        for (ntype, hetero_idx), homo_idx in node_mapping.items():
            type_to_pairs.setdefault(ntype, []).append((hetero_idx, homo_idx))
        result = {}
        for ntype, pairs in type_to_pairs.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            hetero_indices = [p[0] for p in pairs_sorted]
            homo_indices = [p[1] for p in pairs_sorted]
            emb_list = [embeddings_all[h] for h in homo_indices]
            emb_array = np.stack(emb_list, axis=0)
            result[ntype] = {
                'hetero_indices': hetero_indices,
                'embeddings': emb_array,
            }
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Dir with encoder_*.pt and vq_*.pt')
    parser.add_argument('--epoch', type=int, default=18, help='Epoch number to load')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='Root output directory')
    parser.add_argument('--quantized', action='store_true', default=True, help='Use quantized embeddings')
    args = parser.parse_args()

    # Параметры модели – должны совпадать с обучением
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint_dir}, epoch {args.epoch}")
    encoder, vq = load_pretrained_model(params, args.checkpoint_dir, args.epoch, device)

    graph_path = f'data/{args.dataset}/processed/geometric_data_processed.pt'
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    print(f"Loading graph {args.dataset}...")
    data, node_mapping = load_graph_with_mapping(graph_path)

    print("Extracting embeddings...")
    result = extract_embeddings_by_type(encoder, vq, data, node_mapping, device, return_quantized=args.quantized)

    os.makedirs(args.output_dir, exist_ok=True)
    out_npz = os.path.join(args.output_dir, f'{args.dataset}_embeddings.npz')
    out_pkl = os.path.join(args.output_dir, f'{args.dataset}_embeddings.pkl')

    output_data = {}
    for ntype, info in result.items():
        output_data[f'{ntype}_embeddings'] = info['embeddings']
        output_data[f'{ntype}_hetero_indices'] = np.array(info['hetero_indices'])
    np.savez(out_npz, **output_data)
    with open(out_pkl, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved to {args.output_dir}")
    for ntype, info in result.items():
        print(f"Type '{ntype}': {info['embeddings'].shape[0]} nodes, dim {info['embeddings'].shape[1]}")

if __name__ == '__main__':
    main()