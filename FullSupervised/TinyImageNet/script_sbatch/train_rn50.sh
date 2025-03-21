#!/bin/bash -e

#SBATCH --job-name=rn50# create a short name for your job
#SBATCH --output=/lustre/scratch/client/movian/research/users/quanpn2/sbatch_results/mbpp%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/movian/research/users/quanpn2/sbatch_results/mbpp%A.err # create a error file
#SBATCH --partition=movianr # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=64GB
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
export PYTHONPATH=/lustre/scratch/client/movian/research/users/quanpn2/PEFT_HiCE/FullSupervised/TinyImageNet
cd /lustre/scratch/client/movian/research/users/quanpn2/PEFT_HiCE/FullSupervised/TinyImageNet






wandb disabled
# wandb enabled
# wandb online

torchrun --nproc_per_node=1  classification/train.py \
    --model 'resnet50' \
    --batch-size 256 \
    --epochs 50 \
    --opt 'sgd' \
    --lr 0.2 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --lr-scheduler 'cosineannealinglr' \
    --lr-warmup-epochs 5 \
    --lr-warmup-method 'linear' \
    --lr-warmup-decay 0.01 \
    --output-dir 'save/rn50_50ep'
