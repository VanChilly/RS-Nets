"""
[IMPORTANT!!!]

This file is to process input for Resolution Sizes times, i.e., 224, 192, 160, 128, 96, 5 times processing totally.
"""
import torch.nn as nn


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


# private BatchNorm
class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, "bn_" + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, "bn_" + str(i))(x) for i, x in enumerate(x_parallel)]
