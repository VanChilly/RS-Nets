#!/bin/bash

python -u non-quantization/imagenet.py \
    --arch parallel_resnet18 \
    --data /HOME/scz0831/run/prune/dataset/cifar10 \
    --worker 2 \
    --epochs 120 \
    --sizes 224 192 160 128 96 \
    --kd \
    --kd-type ens_topdown > log/log_test_train.log
