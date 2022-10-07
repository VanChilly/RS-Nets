#!/bin/bash

python non-quantization/imagenet.py \
    --arch parallel_resnet18 \
    --data /HOME/scz0831/run/prune/dataset/cifar10 \
    --workers 16 \
    --inference \
    --resume checkpoints/model_best_parallel_resnet18_drs_loss_flops.pth.tar \
    --sizes 224 192 160 128 96 