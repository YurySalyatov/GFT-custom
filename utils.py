import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from typing import Optional

def hetero_to_homo(hetero_data: HeteroData, edge_embeddings_path: Optional[str] = None) -> Data:
    """
    Преобразует гетерогенный граф в однородный с загрузкой эмбеддингов рёбер из файла.

    Параметры:
        hetero_data (HeteroData): исходный гетерогенный граф.
        edge_embeddings_path (str, optional): путь к edge_embeddings.pkl.
            Если None, edge_text_feat заполняется one‑hot матрицей.

    Возвращает:
        Data: объект с полями:
            node_text_feat : эмбеддинги узлов [N, max_dim] (нормализованные)
            edge_text_feat : признаки для каждого типа ребра [num_edge_types, embed_dim]
            x : индексы узлов [N]
            xe : индексы типов рёбер [E]
            edge_index : рёбра [2, E]
            node_type_onehot : one‑hot тип узла [N, num_node_types]
            edge_attr : one‑hot тип ребра [E, num_edge_types]
    """
    import pickle
    excluded_edge_types = [('author', 'knows', 'author')]
    if excluded_edge_types:
        for et in excluded_edge_types:
            if et in hetero_data.edge_types:
                del hetero_data[et]
                print(f"Removed edge type: {et}")
    # 1. Загрузка эмбеддингов рёбер
    edge_emb_dict = None
    embed_dim = None
    if edge_embeddings_path is not None:
        with open(edge_embeddings_path, 'rb') as f:
            edge_emb_dict = pickle.load(f)
        for key in list(edge_emb_dict.keys()):
            emb = edge_emb_dict[key]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float)
            edge_emb_dict[key] = emb
            if embed_dim is None:
                embed_dim = emb.shape[0]
            else:
                assert embed_dim == emb.shape[0], f"Несоответствие размерности для {key}"
        if embed_dim is None:
            raise ValueError("Файл с эмбеддингами пуст.")

    # 2. Обработка узлов (как раньше)
    node_types = hetero_data.node_types
    num_node_types = len(node_types)
    node_dims = {}
    for nt in node_types:
        if hasattr(hetero_data[nt], 'x') and hetero_data[nt].x is not None:
            node_dims[nt] = hetero_data[nt].x.size(1)
        else:
            node_dims[nt] = 0
    max_dim = max(node_dims.values()) if node_dims else 0

    mapping = {}
    global_idx = 0
    all_node_embs, all_node_type_onehot, all_node_indices = [], [], []

    for nt in node_types:
        x = hetero_data[nt].x if hasattr(hetero_data[nt], 'x') else None
        num_nodes = hetero_data[nt].num_nodes

        if x is not None:
            x_norm = F.normalize(x, p=2, dim=1)
            if x_norm.size(1) < max_dim:
                pad = torch.zeros(x_norm.size(0), max_dim - x_norm.size(1))
                x_pad = torch.cat([x_norm, pad], dim=1)
            else:
                x_pad = x_norm
        else:
            x_pad = torch.zeros(num_nodes, max_dim)

        all_node_embs.append(x_pad)

        one_hot = F.one_hot(torch.full((num_nodes,), node_types.index(nt), dtype=torch.long),
                            num_classes=num_node_types).float()
        all_node_type_onehot.append(one_hot)

        indices = torch.arange(global_idx, global_idx + num_nodes)
        all_node_indices.append(indices)

        for local_idx in range(num_nodes):
            mapping[(nt, local_idx)] = global_idx
            global_idx += 1

    node_text_feat = torch.cat(all_node_embs, dim=0)
    node_type_onehot = torch.cat(all_node_type_onehot, dim=0)
    x = torch.cat(all_node_indices, dim=0)

    # 3. Обработка рёбер
    edge_types = hetero_data.edge_types
    num_edge_types = len(edge_types)

    # Формируем edge_text_feat
    if edge_emb_dict is not None:
        edge_emb_list = []
        for et in edge_types:
            emb = edge_emb_dict.get(et)
            if emb is None:
                print(f"Внимание: не найден эмбеддинг для {et}. Используется нулевой вектор.")
                emb = torch.zeros(embed_dim)
            edge_emb_list.append(emb)
        edge_text_feat = torch.stack(edge_emb_list, dim=0)  # [num_edge_types, embed_dim]
    else:
        edge_text_feat = torch.eye(num_edge_types)
    all_edge_indices, all_edge_attr, all_xe = [], [], []
    for idx_et, et in enumerate(edge_types):
        src_type, _, dst_type = et
        edge_index = hetero_data[et].edge_index
        num_edges = edge_index.size(1)

        src_local = edge_index[0]
        dst_local = edge_index[1]
        src_global = torch.tensor([mapping[(src_type, int(i))] for i in src_local])
        dst_global = torch.tensor([mapping[(dst_type, int(i))] for i in dst_local])
        edge_global = torch.stack([src_global, dst_global], dim=0)
        all_edge_indices.append(edge_global)

        edge_one_hot = F.one_hot(torch.full((num_edges,), idx_et, dtype=torch.long),
                                 num_classes=num_edge_types).float()
        all_edge_attr.append(edge_one_hot)

        xe_local = torch.full((num_edges,), idx_et, dtype=torch.long)
        all_xe.append(xe_local)

    if all_edge_indices:
        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
        xe = torch.cat(all_xe, dim=0)
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.float)
        xe = torch.tensor([], dtype=torch.long)

    # 4. Сборка Data
    data = Data(
        node_text_feat=node_text_feat,
        edge_text_feat=edge_text_feat,
        x=x,
        xe=xe,
        edge_index=edge_index,
        node_type_onehot=node_type_onehot,
        edge_attr_onehot=edge_attr,
        node_mapping=mapping
    )
    return data