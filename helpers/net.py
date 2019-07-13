import torch.nn as nn
import torchvision.models as models
from torch.functional import F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.softmax = F.softmax
        self.log_softmax = F.log_softmax
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.resnet34 = models.resnet101(pretrained=True)

        # with open('dummy.txt', 'w') as f:
        #     f.write(str(list(self.resnet34.children())))
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])

        # self.resnet34.fc = nn.Linear(512, 15)
        self.final_conv = nn.Conv2d(2048, 15, 1)

    def forward(self, input):
        x = self.resnet34(input)
        x = self.global_max_pool(x)
        x = self.final_conv(x)
        x = self.log_softmax(x.squeeze())
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # self.relu = F.relu
        self.relu = F.selu
        self.softmax = F.softmax
        self.log_softmax = F.log_softmax

        self.max_pool = nn.MaxPool2d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3)
        self.bnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 32, 3)
        self.bnorm3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, 3)
        self.bnorm4 = nn.BatchNorm2d(32)

        self.final_conv1 = nn.Conv2d(32, 15, 1)
        self.final_conv2 = nn.Conv2d(15, 15, 4)

    def forward(self, input):
        # x = self.max_pool(self.relu(self.bnorm1(self.conv1(input))))
        #
        # x = self.max_pool(self.relu(self.bnorm2(self.conv2(x))))
        #
        # x = self.max_pool(self.relu(self.bnorm3(self.conv3(x))))
        #
        # x = self.max_pool(self.relu(self.bnorm4(self.conv4(x))))


        x = self.max_pool(self.relu(self.conv1(input)))

        x = self.max_pool(self.relu(self.conv2(x)))

        x = self.max_pool(self.relu(self.conv3(x)))

        # x = self.max_pool(self.relu(self.conv4(x)))

        # x = self.max_pool(self.activation(self.final_conv(x)))

        x = self.relu(self.final_conv1(x))
        # x = self.softmax(self.final_conv2(x))
        # print(x.size())

        x = self.log_softmax(self.global_max_pool(x))
        return x.squeeze()