import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.drop(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()
        self.dataset = dataset
        if self.dataset.startswith('liver_ct'):
            self.inplanes = 16
            print(bottleneck)
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2)
            self.avgpool = nn.AvgPool2d(8)
            # self.fc = nn.Linear(64 * block.expansion, num_classes)
            self.fc = nn.Linear(1024 * block.expansion, num_classes)
            self.drop = nn.Dropout(p=0.5)

        # elif dataset == 'imagenet':
        else:
            print('here')
            blocks = {10: BasicBlock, 18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                      200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.inplanes = 32
            self.plane = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], self.plane, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 2*self.plane, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 4*self.plane, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 8*self.plane, layers[depth][3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            # self.avgpool = nn.AvgPool2d(2)
            # self.fc = nn.Linear(2048, num_classes)
            self.fc = nn.Linear(256 * blocks[depth].expansion, num_classes)
            self.fc1=nn.Sequential(nn.Linear(8*self.plane*blocks[depth].expansion,2*self.plane,bias=False),nn.ReLU(inplace=True))
            self.fc2=nn.Linear(2*self.plane,num_classes,bias=False)


            self.drop = nn.Dropout(p=0.5)

            self.softmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'liver_ct' or self.dataset == 'liver_mr':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.drop(x)


            x = self.layer1(x)
            x = self.drop(x)
            x = self.layer2(x)
            x = self.drop(x)
            x = self.layer3(x)
            x = self.drop(x)

            x = self.avgpool(x)
            # print('avgpool: ', x.size())
            x = x.view(x.size(0), -1)
            x = self.drop(x)
            # print('x: ', x.size())
            x = torch.sigmoid(self.fc(x))


        # elif self.dataset == 'imagenet':
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            # x = self.drop(x)

            x = self.layer1(x)
            # x = self.drop(x)
            x = self.layer2(x)
            # x = self.drop(x)
            x = self.layer3(x)
            # x = self.drop(x)
            x = self.layer4(x)
            x = self.drop(x)
            # print(x.size())

            x = self.avgpool(x)
            # print(x.size())
            x = x.view(x.size(0), -1)
            # print(x.size())
            # x = self.drop(x)
            # print(x.size())
            #x = torch.sigmoid(self.fc(x))
            # x = self.softmax(self.fc(x))
            x = self.fc2(self.fc1(x))

        return x

if __name__ == "__main__":
    num_classes = 8
    input_tensor = torch.autograd.Variable(torch.rand(32, 3, 224, 224))
    # model = resnet50(class_num=num_classes)
    model = ResNet(dataset = 'calc', depth = 34, num_classes=num_classes)
    output = model(input_tensor)

    print(output)