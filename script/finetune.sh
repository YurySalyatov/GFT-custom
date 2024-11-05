#!/bin/bash

#$ -M zwang43@nd.edu
#$ -m abe
#$ -N Finetune
#$ -q gpu@@yye7_lab
#$ -pe smp 8
#$ -l gpu=1

for dataset in cora pubmed arxiv wikics WN18RR FB15K237 chemhiv chempcba
do
    ~/.conda/envs/GFT/bin/python GFT/finetune.py --use_params --dataset $dataset
done