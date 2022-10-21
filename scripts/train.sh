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
python -u non-quantization/imagenet.py \
    --arch parallel_resnet18 \
    --data /HOME/scz0831/run/prune/dataset/ImageNet100 \
    --worker 16 \
    --epochs 120 \
    --train_mode 1 \
    --sizes 224 \
    > log/log.log

# for training DRS
# python -u non-quantization/imagenet.py \
#     --arch parallel_resnet18 \
#     --data /HOME/scz0831/run/prune/dataset/ImageNet100 \
#     --resume checkpoints/resnet18_3_reso/parallel_resnet18_ImageNet100_3_reso_backbone_1.pth.tar \
#     --worker 16 \
#     --train_mode 2 \
#     --epochs 20 \
#     --sizes 224 192 160 \
#     --flops_loss DRNet \
#     --eta 0.1 \
#     --alpha 1.36 \
#     > log/log.log
