# -*- coding:utf-8 -*-  
'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# 定义是否使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数设置
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_loss_everyepoch = 0
train_accuracy_everyepoch = 0
test_loss_everyepoch = 0
test_accuracy_everyepoch = 0


# Data
print('==> Preparing data..')
# torchvision.transforms是pytorch中的图像预处理包 一般用Compose把多个步骤整合到一起：
transform_train = transforms.Compose([  # 通过compose将各个变换串联起来
    transforms.RandomCrop(32, padding=4),   # 先四周填充0，再把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),      # 以0.5的概率水平翻转给定的PIL图像
    # transforms.RandomAffine(5.0),         # python2.7 没有
    transforms.RandomGrayscale(p=0.1),   # 依概率 p 将图片转换为灰度图
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


## if you cannot run the program because of "OUT OF MEMORY", you can decrease the batch_size properly.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

# DataLoader接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
# 生成一个个batch进行批训练，组成batch的时候顺序打乱取
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

print('SAMPLES:', len(trainset), len(testset))
print('EPOCH:', len(trainloader), len(testloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # 类型

# Model
print('==> Building model..')

# 定义卷积神经网络
from DaiNet import *

net = DaiNet()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True


# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(os.path.join('./checkpoint', 'DaiNet', 'ckpt.t7'))
#checkpoint = torch.load('./checkpoint/ckpt.t7')
net.load_state_dict(checkpoint['net'])
net.fc1 = nn.Sequential()

print(net)

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            print(inputs.shape)
            outputs = net(inputs)
            # print(outputs.shape)

# 测试
test(epoch = 1)
