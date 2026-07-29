import torch
import os
graph_dir = f"{os.getcwd()}/../graph_dir/"
print(os.getcwd() + graph_dir)
names = ['arnetminer', 'qian', 'zbmath']
from utils import hetero_to_homo

for name in names:
    # hetero_data = torch.load(f"{graph_dir}{name}/hetero_graph.pt", weights_only=False)
    hetero_data = torch.load(f"{graph_dir}{name}/new_subgraph_with_max_15_duplicates_2.pt", weights_only=False)
    data = hetero_to_homo(hetero_data, f"{graph_dir}edge_embeddings.pkl")
    print(data)
    # convert = convert_to_gft_format(data)
    save_path = f"data/{name}/processed"
    os.makedirs(save_path, exist_ok=True)
    torch.save(data, f"{save_path}/geometric_data_processed.pt")