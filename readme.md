# Foundation Graph Model with Tree Vocabulary

<img src="assets/framework.png">

During pre-training, the model encodes general knowledge from a graph database into a tree vocabulary through tree
reconstruction. In fine-tuning, the learned tree vocabulary is applied to unify graph-related tasks as tree
classification, adapting the general knowledge to specific tasks.

## Getting Started

### Setup Environment

```
conda env create -f environment.yml
conda activate GFM
```

### Pretraining

For pre-train the model, please run the commend:

```
cd pretrain/
python pretrain.py
```

We have set the essential pre-training hyper-parameters. You can just run the above script directly. The parameters
would be stored in `/params/pretrain_model/`

### Fine-tuning

Run the command for pre-training and fine-tuning:

```
cd finetune/
python finetune.py --dataset cora --update_init_params
```

Run the command for few-shot learning:

```
python finetune.py --dataset cora --update_init_params --setting few_shot --n_way 5 --n_shot 3
```

You can run the experiments on cora, pubmed, wikics, arxiv, WN18RR, FB15K237, chemhiv, chempcba. 