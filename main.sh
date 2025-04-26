wandb login

python3 src/train_model.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 1 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "dinov2_upsample" \
    --wandb-save True \