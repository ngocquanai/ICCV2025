#!/bin/bash -e

#SBATCH --job-name=ls# create a short name for your job
#SBATCH --output=/lustre/scratch/client/movian/research/users/quanpn2/sbatch_results/mbpp%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/movian/research/users/quanpn2/sbatch_results/mbpp%A.err # create a error file
#SBATCH --partition=movianr # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.quanpn2@vinai.io



module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"



conda activate peft
export PYTHONPATH=/lustre/scratch/client/movian/research/users/quanpn2/PEFT_HiCE/ImageClassification






model=vit_base_patch16_224_in21k_vptshallow
model_type=vit_vptshallow
model_checkpoint=./ViT-B_16.npz
topN=100
tuning_mode=vptshallow

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset oxford_iiit_pet --task vtab --lr 0.0007 --wd 0.8 --config_folder configs_LS --output output_ls --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset sun397 --task vtab --lr 0.0017 --wd 0.01 --config_folder configs_LS --output output_ls --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN
