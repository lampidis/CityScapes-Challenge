wandb login

python3 src/my_dinov2.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 60 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2_upsample" \
    --wandb-save True \