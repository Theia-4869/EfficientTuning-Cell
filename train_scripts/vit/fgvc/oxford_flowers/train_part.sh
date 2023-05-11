gpu_id=1
bz=32
lr=0.0008

python train_our.py /data/zqz/FGVC \
    --dataset oxford_flowers \
    --num-classes 102 --simple-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam  --weight-decay 0.0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
    --model-ema --model-ema-decay 0.999 \
    --output output/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 2 \
    --gpu-id $gpu_id \
    --log-wandb \
    --experiment flowers_zqz \
    --run-name test \
    --contrast-aug --no-prefetcher --contrastive
