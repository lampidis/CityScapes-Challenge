wandb login

python3 my_dinov2.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 60 \
    --lr 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2_lrs" \
    --wandb-save True \