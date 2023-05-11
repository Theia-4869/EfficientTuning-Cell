gpu_id=1
bz=64
lr=0.001

python train_our.py /data/zqz/data/FGVC \
    --dataset nabirds \
    --num-classes 555 --simple-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam  --weight-decay 0.0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
    --model-ema --model-ema-decay 0.9998 \
    --output output/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 2 \
    --gpu_id $gpu_id \
    --log-wandb \
    --experiment nabirds_zqz \
    --run_name test \
    --contrast-aug --no-prefetcher --contrastive
