model=vit_base_patch16_224_in21k_lora
model_type=vit_lora
model_checkpoint=/lustre/scratch/client/movian/research/users/quanpn2/Parameter-Efficient-Transfer-Learning-Benchmark/ImageClassification/ViT-B_16.npz
topN=8


CUDA_VISIBLE_DEVICES=0 python train/train_model_lora.py --dataset resisc45 --task vtab --lr 0.005 --wd 0.01 --eval True --dpr 0.1 --tuning_mode lora --model_type $model_type --model $model --model_checkpoint $model_checkpoint --topN $topN




