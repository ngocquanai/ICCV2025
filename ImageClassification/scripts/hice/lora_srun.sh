
model=vit_base_patch16_224_in21k_lora
model_type=vit_lora
model_checkpoint=/lustre/scratch/client/movian/research/users/quanpn2/Parameter-Efficient-Transfer-Learning-Benchmark/ImageClassification/ViT-B_16.npz
topN=8


CUDA_VISIBLE_DEVICES=0 python train/train_model_lora.py --dataset cifar100 --task vtab --lr 0.005 --wd 0.005 --config_folder configs_HiCE --output output_hicea --eval True --dpr 0.1 --tuning_mode lora --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_lora.py --dataset caltech101 --task vtab --lr 0.006 --wd 0.35 --config_folder configs_HiCE --output output_hicea --eval True --dpr 0.1 --tuning_mode lora --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_lora.py --dataset oxford_iiit_pet --task vtab --lr 0.0007 --wd 0.8 --config_folder configs_HiCE --output output_hicea --eval True --dpr 0.1 --tuning_mode lora --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_lora.py --dataset sun397 --task vtab --lr 0.0017 --wd 0.01 --config_folder configs_HiCE --output output_hicea --eval True --dpr 0.1 --tuning_mode lora --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN

CUDA_VISIBLE_DEVICES=0 python train/train_model_lora.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.005 --config_folder configs_HiCE --output output_hicea --eval True --dpr 0.1 --tuning_mode lora --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN
