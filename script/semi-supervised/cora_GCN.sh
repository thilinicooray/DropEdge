#!/bin/bash

python ./src/train_new.py \
    --debug \
    --datapath data// \
    --seed 42 \
    --dataset cora \
    --type resgcn \
    --nhiddenlayer 2 \
    --nbaseblocklayer 0 \
    --hidden 128 \
    --epoch 400 \
    --lr 0.007 \
    --weight_decay 1e-05 \
    --early_stopping 400 \
    --sampling_percent 1.0 \
    --dropout 0.6 \
    --normalization AugNormAdj --task_type semi \
     \
    
