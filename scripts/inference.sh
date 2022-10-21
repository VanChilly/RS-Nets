#!/bin/bash

python non-quantization/imagenet.py \
    --arch parallel_resnet18 \
    --data /HOME/scz0831/run/prune/dataset/ImageNet100 \
    --workers 16 \
    --inference \
    --resume checkpoints/parallel_resnet18_ImageNet100_3_reso_drs_0.1_1.pth.tar \
    --sizes 224 192 160