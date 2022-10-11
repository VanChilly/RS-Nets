#!/bin/bash

# for ens-topdown kd
# python -u non-quantization/imagenet.py \
#     --arch parallel_resnet18 \
#     --data /HOME/scz0831/run/prune/dataset/cifar10 \
#     --worker 2 \
#     --epochs 120 \
#     --sizes 224 192 160 128 96 \
#     --kd \
#     --kd-type ens > log/log.log

# for ens kd
# python -u non-quantization/imagenet.py \
#     --arch parallel_resnet18 \
#     --data /HOME/scz0831/run/prune/dataset/cifar10 \
#     --worker 2 \
#     --epochs 120 \
#     --sizes 224 192 160 128 96 \
#     --kd \
#     --kd-type ens > log/log.log

# for no kd low reso
# python -u non-quantization/imagenet.py \
#     --arch parallel_resnet18 \
#     --data /HOME/scz0831/run/prune/dataset/cifar10 \
#     --worker 2 \
#     --epochs 120 \
#     --train_mode 1 \
#     --sizes 32 26 20 14 8 \
#     > log/log.log

# for training DRS
python -u non-quantization/imagenet.py \
    --arch parallel_resnet18 \
    --data /HOME/scz0831/run/prune/dataset/cifar10 \
    --resume checkpoints/checkpoint_model_best_parallel_resnet18_no_kd_low_reso.pth.tar \
    --worker 2 \
    --train_mode 2 \
    --epochs 120 \
    --sizes 32 26 20 14 8 \
    --flops_loss \
    # > log/log.log
