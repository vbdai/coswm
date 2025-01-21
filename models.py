import torch.nn as nn
import torch.nn.functional as F


# ResNet model classes

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 1. residual
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * BasicBlock.expansion)

        # 2. shortcut
        self.shortcut = nn.Sequential()

        # the shorcut dimension is not the same with residual
        # use 1*1 convolution to match
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleNeck, self).__init__()
        # 1. residual
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * BottleNeck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * BottleNeck.expansion)

        # 2. shortcut
        self.shortcut = nn.Sequential()

        # the shorcut dimension is not the same with residual
        # use 1*1 convolution to match
        if stride != 1 or in_planes != planes * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BottleNeck.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels=3, num_classes=10):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer_(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer_(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # [stride, 1, 1, ..., 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 3 -> 64
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # blocks
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # to linear
        out = self.linear(out)
        return out


def ResNet18(num_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_channels, num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=10):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=10):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)
