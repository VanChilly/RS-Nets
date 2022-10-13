#!/bin/bash

python non-quantization/imagenet.py \
    --arch parallel_resnet20 \
    --data /HOME/scz0831/run/prune/dataset/cifar10 \
    --workers 16 \
    --inference \
    --resume checkpoints/model_best.pth.tar \
    --sizes 32 26 20