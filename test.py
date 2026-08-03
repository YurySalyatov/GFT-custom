import torch
import numpy as np

# names = ["chemblpre", "chemhiv", "chempcba", "Cora", "FB15K237", "Pubmed", "wikics", "WN18RR"]
names = ['Pubmed']
for name in names:
    print(name)
    data = torch.load(f'GFT-data/GFT-data/{name}/processed/data.pt', weights_only=False)
    print("data:\n", data)
    geometric_data = torch.load(f'GFT-data/GFT-data/{name}/processed/geometric_data_processed.pt', weights_only=False)
    print("geometric data:\n", geometric_data)
    pre_filter = torch.load(f'GFT-data/GFT-data/{name}/processed/pre_filter.pt', weights_only=False)
    print("pre filter:\n", pre_filter)
    pre_transform = torch.load(f'GFT-data/GFT-data/{name}/processed/pre_transform.pt', weights_only=False)
    print("pre transform\n", pre_transform)
    texts = np.load(f'GFT-data/GFT-data/{name}/processed/texts.pkl')
    print("texts:\n", texts)