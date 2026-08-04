import torch
import os
import argparse

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--duplicate', action='store_true',
                    help='If True, use new_subgraph_with_max_15_duplicates_2.pt, otherwise use hetero_graph.pt')
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
    if args.duplicate:
        filename = "new_subgraph_with_max_15_duplicates_2.pt"
    else:
        filename = "hetero_graph.pt"

    hetero_data = torch.load(f"{graph_dir}{name}/{filename}", weights_only=False)
    data = hetero_to_homo(hetero_data, f"{graph_dir}edge_embeddings.pkl")

    save_path = f"data/{name}/processed"
    os.makedirs(save_path, exist_ok=True)
    torch.save(data, f"{save_path}/geometric_data_processed.pt")