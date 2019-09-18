import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math

class AlexNetFc(nn.Module):
  def __init__(self, hash_bit):
    super(AlexNetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True) # 预训练模型
    self.features = model_alexnet.features  # features里边包含了12个子层  可以print查看
    self.classifier = nn.Sequential()
    for i in range(6):  # 载入前五层
        self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    
    # 对于HashNet来说 原来的features层和classifier层表示为  新的feature_layers
    self.feature_layers = nn.Sequential(self.features, self.classifier)  
    
    #..................# 定义hash_layer
    self.use_hashnet = True   #use_hashnet
    self.hash_layer = nn.Linear(model_alexnet.classifier[6].in_features, hash_bit)  # 全连接层转化为hash_bit K-dimensional representation
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh() # tanh(\beta z)
    self.scale = self.init_scale

  def forward(self, x):
    if self.training: # 每进行一次前项传播 迭代次数计数+1
        self.iter_num += 1
    x = self.features(x)
    # print(x.size(0)) 输出的值为72
    x = x.view(x.size(0), 256*6*6)  # 转化成一个 指定行列的向量  
    # print(x.size(0),x.size(1))  72 9216
    x = self.classifier(x)
    y = self.hash_layer(x)  # 72 x 48
    if self.iter_num % self.step_size==0:   # 每200步 \beta 按公式缩小一次
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power)) 
    y = self.activation(self.scale*y) # y = sgn(y) = tanh(\beta z)
    return y

  def output_num(self): # 返回hash_bit
    return self.__in_features

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152} 
class ResNetFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn} 
class VGGFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_hashnet = use_hashnet
    self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    x = self.features(x)
    x = x.view(x.size(0), 25088)
    x = self.classifier(x)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features
