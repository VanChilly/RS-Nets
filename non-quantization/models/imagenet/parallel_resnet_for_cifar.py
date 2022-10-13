import torch.nn as nn
from .parallel import ModuleParallel, BatchNorm2dParallel

__all__ = ['parallel_resnet20']

def conv3x3(inplanes, planes, stride=1):
    return ModuleParallel(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, 
    stride=stride, padding=1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_parallel=3):
        super().__init__()
        self.norm_layer = BatchNorm2dParallel
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.norm_layer(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = self.norm_layer(planes, num_parallel)
        self.downsample = downsample
        self.num_parallel = num_parallel
        self.stride = stride

    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = [out[i] + identity[i] for i in range(self.num_parallel)]
        out = self.relu(out)

        return out
    

class Parallel_ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10, num_parallel=3) -> None:
        super().__init__()
        self.in_channels = 16
        self.num_parallel = num_parallel
        self.conv1 = ModuleParallel(nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False))
        self.bn = BatchNorm2dParallel(16, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)

        self.avg_pool = ModuleParallel(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = ModuleParallel(nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            self.downsample = nn.Sequential()
            if stride != 1 or self.in_channels != out_channels:
                self.downsample = nn.Sequential(
                    ModuleParallel(nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                    BatchNorm2dParallel(out_channels, self.num_parallel)
                )
            layers.append(block(self.in_channels, out_channels, stride, self.downsample, self.num_parallel))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.relu(self.bn(self.conv1(x)))

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)

        out = self.avg_pool(out)

        out = [t.reshape(t.size(0), -1) for t in out]
        out = self.fc(out)

        return out

def parallel_resnet20(num_parallel=3, num_classes=10):
    return Parallel_ResNet(BasicBlock, [3, 3, 3], num_classes, num_parallel)


if __name__ == '__main__':
    import torch
    model = parallel_resnet20(10)
    x = torch.randn((1, 3, 32, 32))
    print(model(x).shape)