import torch
import torch.nn.functional as F
import os
import argparse

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='arnetminer,qian,zbmath',
                    help='Comma-separated list of dataset names, e.g., "arnetminer,qian"')
args = parser.parse_args()

graph_dir = f"{os.getcwd()}/graph_dir/"
print(os.getcwd() + graph_dir)

# Разбиваем строку на список, удаляем лишние пробелы
names = [name.strip() for name in args.datasets.split(',') if name.strip()]

from utils import hetero_to_homo

for name in names:
    # Выбор имени файла в зависимости от флага duplicate
    filename = "homo_graph_with_embeddings.pt"

    homo_data = torch.load(f"{graph_dir}{name}/{filename}", weights_only=False)
    print(homo_data["node_text_feat"][0])
    # Нормализация эмбеддингов узлов
    if "node_text_feat" in homo_data:
        if isinstance(homo_data["node_text_feat"], list):
            for i, n_emb in enumerate(homo_data["node_text_feat"]):
                homo_data["node_text_feat"][i] = F.normalize(n_emb, p=2, dim=-1)
            homo_data["node_text_feat"] = torch.tensor(homo_data["node_text_feat"], dtype=torch.float32)
        else:
            homo_data["node_text_feat"] = F.normalize(homo_data["node_text_feat"], p=2, dim=-1)
    print(homo_data["node_text_feat"][0])

    print(homo_data["edge_text_feat"][0])
    # Нормализация эмбеддингов рёбер
    if "edge_text_feat" in homo_data:
        if isinstance(homo_data["edge_text_feat"], list):
            for i, edge_emb in enumerate(homo_data["edge_text_feat"]):
                homo_data["edge_text_feat"][i] = F.normalize(edge_emb, p=2, dim=-1)
            homo_data["edge_text_feat"] = torch.tensor(homo_data["node_text_feat"], dtype=torch.float32)
        else:
            homo_data["edge_text_feat"] = F.normalize(homo_data["edge_text_feat"], p=2, dim=-1)
    print(homo_data["edge_text_feat"][0])

    save_path = f"data/{name}/processed"
    os.makedirs(save_path, exist_ok=True)
    torch.save(homo_data, f"{save_path}/geometric_data_processed.pt")