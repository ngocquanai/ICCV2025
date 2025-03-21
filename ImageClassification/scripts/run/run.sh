
model=vit_base_patch16_224_in21k_adapter
model_type=vit_adapter
model_checkpoint=./ViT-B_16.npz
topN=96

export PYTHONPATH=/lustre/scratch/client/movian/research/users/quanpn2/PEFT_HiCE/ImageClassification

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset cifar100 --task vtab --lr 0.01 --wd 1.0 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset caltech101 --task vtab --lr 0.006 --wd 0.35 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset oxford_iiit_pet --task vtab --lr 0.0007 --wd 0.8 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset oxford_flowers102 --task vtab --lr 0.05 --wd 0.11 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset sun397 --task vtab --lr 0.0017 --wd 0.01 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.005 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset cifar100 --task vtab --lr 0.01 --wd 1.0 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset caltech101 --task vtab --lr 0.006 --wd 0.35 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset oxford_iiit_pet --task vtab --lr 0.0007 --wd 0.8 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_adapter.py --dataset oxford_flowers102 --task vtab --lr 0.05 --wd 0.11 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset sun397 --task vtab --lr 0.0017 --wd 0.01 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_adapter.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.005 --config_folder configs_HiCE --output output_debug --eval True --dpr 0.1 --tuning_mode adapter --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN


