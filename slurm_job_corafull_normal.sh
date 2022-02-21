#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --output=../slurm_output/gpn-%N-%j  # %N for node name, %j for jobID
#SBATCH --mem=64G

cd ~/Graph-Posterior-Network

#python3 train_and_eval.py with configs/gpn/file-dataset/CoraFullMLvsOS/loc-hybrid.yaml
#python3 train_and_eval.py with configs/gpn/file-dataset/CoraFullMLvsOS/loc-transductive.yaml

#python3 train_and_eval.py with configs/gpn/file-dataset/CoraFullMLvsOS/bernoulli-hybrid.yaml
#python3 train_and_eval.py with configs/gpn/file-dataset/CoraFullMLvsOS/bernoulli-transductive.yaml

#python3 train_and_eval.py with configs/gpn/file-dataset/CoraFullMLvsOS/normal-hybrid.yaml
python3 train_and_eval.py with configs/gpn/file-dataset/CoraFullMLvsOS/normal-transductive.yaml