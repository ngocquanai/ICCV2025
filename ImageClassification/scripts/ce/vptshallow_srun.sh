
model=vit_base_patch16_224_in21k_vptshallow
model_type=vit_vptshallow
model_checkpoint=./ViT-B_16.npz
topN=100
tuning_mode=vptshallow

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset cifar100 --task vtab --lr 1e-3 --wd 1e-4 --config_folder configs_CE --output output_ce --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset caltech101 --task vtab --lr 0.006 --wd 0.35 --config_folder configs_CE --output output_ce --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN


CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset oxford_iiit_pet --task vtab --lr 0.0007 --wd 0.8 --config_folder configs_CE --output output_ce --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_vptshallow.py --dataset oxford_flowers102 --task vtab --lr 0.05 --wd 0.11 --config_folder configs_CE --output output_ce --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset sun397 --task vtab --lr 0.0017 --wd 0.01 --config_folder configs_CE --output output_ce --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_vptshallow.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.01 --config_folder configs_CE --output output_ce --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN
