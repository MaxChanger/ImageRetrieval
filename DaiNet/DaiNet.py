import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class DaiNet(nn.Module): # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(DaiNet, self).__init__()      # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1)
            # nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.Dropout(0.5),
            # nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            # nn.MaxPool2d(2, 2)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # nn.Dropout(0.5),
            nn.AvgPool2d(8, 8)
        )
        
        self.fc1 =  nn.Sequential(
            nn.Linear(1 * 1 * 256, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # fully connect
        x = x.view(-1, 1 * 1 * 256)  

        x = self.fc1(x)
        return x

def dainet():
    return DaiNet()