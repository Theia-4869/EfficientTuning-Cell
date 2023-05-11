gpu_id=1
bz=64
lr=0.003

python train_our.py /data/zqz/FGVC \
    --dataset stanford_cars \
    --num-classes 196 --simple-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam  --weight-decay 0.0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
    --model-ema --model-ema-decay 0.9998 \
    --output output/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 2 \
    --gpu-id $gpu_id \
    --log-wandb \
    --experiment cars_zqz \
    --run-name test \
    --contrast-aug --no-prefetcher --contrastive
