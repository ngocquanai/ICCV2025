#!/bin/bash -e

#SBATCH --job-name=hice# create a short name for your job
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

model=vit_base_patch16_224_in21k_lora
model_type=vit_lora
model_checkpoint=/lustre/scratch/client/movian/research/users/quanpn2/Parameter-Efficient-Transfer-Learning-Benchmark/ImageClassification/ViT-B_16.npz
topN=8


CUDA_VISIBLE_DEVICES=0 python train/train_model_lora.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.005 --config_folder configs_HiCE --output output_hice --eval True --dpr 0.1 --tuning_mode lora --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN





model=vit_base_patch16_224_in21k_adapter
model_type=vit_adapter
model_checkpoint=./ViT-B_16.npz
topN=96

export PYTHONPATH=/lustre/scratch/client/movian/research/users/quanpn2/PEFT_HiCE/ImageClassification

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.005 --config_folder configs_HiCE --output output_hice --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN



model=vit_base_patch16_224_in21k_fclayer
model_type=vit_fclayer
model_checkpoint=./ViT-B_16.npz
topN=8
tuning_mode=fclayer

CUDA_VISIBLE_DEVICES=0 python train/train_model_fclayer.py --dataset cifar100 --task vtab --lr 1e-2 --wd 1e-4 --config_folder configs_HiCE --output output_hice --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_fclayer.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.005 --config_folder configs_HiCE --output output_hice --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN











model=vit_base_patch16_224_in21k_vptshallow
model_type=vit_vptshallow
model_checkpoint=./ViT-B_16.npz
topN=100
tuning_mode=vptshallow

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset cifar100 --task vtab --lr 5e-2 --wd 1e-4 --config_folder configs_HiCE --output output_hice --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.01 --config_folder configs_HiCE --output output_hice --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN
