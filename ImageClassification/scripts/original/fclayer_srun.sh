
model=vit_base_patch16_224_in21k_fclayer
model_type=vit_fclayer
model_checkpoint=./ViT-B_16.npz
topN=8
tuning_mode=fclayer

CUDA_VISIBLE_DEVICES=0 python train/train_model_fclayer.py --dataset cifar100 --task vtab --lr 1e-3 --wd 1e-4 --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_fclayer.py --dataset caltech101 --task vtab --lr 0.006 --wd 0.35 --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN


CUDA_VISIBLE_DEVICES=0 python train/train_model_fclayer.py --dataset oxford_iiit_pet --task vtab --lr 0.0007 --wd 0.8 --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train_model_fclayer.py --dataset oxford_flowers102 --task vtab --lr 0.05 --wd 0.11 --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_fclayer.py --dataset sun397 --task vtab --lr 0.0017 --wd 0.01 --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_fclayer.py --dataset resisc45 --task vtab --lr 0.01 --wd 0.005 --eval True --dpr 0.1 --tuning_mode $tuning_mode --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN
