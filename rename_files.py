import torch
import os
import argparse

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='arnetminer,qian,zbmath',
                    help='Comma-separated list of dataset names, e.g., "arnetminer,qian"')
args = parser.parse_args()

graph_dir = f"{os.getcwd()}/graph_dir2/"
print(os.getcwd() + graph_dir)

# Разбиваем строку на список, удаляем лишние пробелы
names = [name.strip() for name in args.datasets.split(',') if name.strip()]

from utils import hetero_to_homo

for name in names:
    # Выбор имени файла в зависимости от флага duplicate
    filename = "homo_graph_with_embeddings.pt"

    homo_data = torch.load(f"{graph_dir}{name}/{filename}", weights_only=False)
    save_path = f"data/{name}/processed"
    os.makedirs(save_path, exist_ok=True)
    torch.save(homo_data, f"{save_path}/geometric_data_processed.pt")