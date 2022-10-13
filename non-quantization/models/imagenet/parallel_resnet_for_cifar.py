import torch.nn as nn
from .parallel import ModuleParallel, BatchNorm2dParallel

def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, 
    stride=stride, padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
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
        
        out += identity
        out = self.relu(out)

        return out
    

class Parallel_ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=10) -> None:
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
            self.shortcut = nn.Sequential()
            if stride != 1 or self.in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            layers.append(block(self.in_channels, out_channels, stride, self.shortcut))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)

        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def resnet20(num_classes=10, **kwargs):
    return Parallel_ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


if __name__ == '__main__':
    import torch
    model = resnet20(10)
    x = torch.randn((1, 3, 32, 32))
    print(model(x).shape)