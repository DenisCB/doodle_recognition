import torch.nn as nn
import numpy as np
from ML.torch_blocks import conv_layer, linear_layer, InceptionA, InceptionC


class ConvNetSimple(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(ConvNetSimple, self).__init__()
        self.layer1 = conv_layer(input_dim, 128, 5, 1, 0.05, 2)
        self.layer2 = conv_layer(self.layer1.out_dim, 128, 3, 1, 0.05, 2)
        self.layer3 = conv_layer(self.layer2.out_dim, 256, 3, 1, 0.05, 2)
        self.layer4 = conv_layer(self.layer3.out_dim, 128, 1, 1, 0.05, 1)
        self.layer5 = conv_layer(self.layer4.out_dim, 256, 3, 1, 0.05, 2)
        self.layer6 = conv_layer(self.layer5.out_dim, 256, 3, 1, 0.05, 1)

        num_features = np.prod(self.layer6.out_dim)
        self.linear1 = linear_layer(num_features, 128, True, True, 0.05)
        self.linear2 = nn.Linear(
            in_features=self.linear1.out_dim,
            out_features=num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class InceptionBasedNet(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(InceptionBasedNet, self).__init__()
        self.layer1 = conv_layer(input_dim, 128, 5, 1, 0.05, 2)

        self.inc_A1 = InceptionA(self.layer1.out_dim, 32)
        self.inc_A2 = InceptionA(self.inc_A1.out_dim, 64)

        self.inc_C1 = InceptionC(self.inc_A2.out_dim, 64)
        self.inc_C2 = InceptionC(self.inc_C1.out_dim, 64)

        self.layer2 = conv_layer(self.inc_C2.out_dim, 128, 3, 1, 0.05, 2)

        num_features = np.prod(self.layer2.out_dim)
        self.linear1 = linear_layer(num_features, 64, True, True, 0.05)
        self.linear2 = nn.Linear(in_features=self.linear1.out_dim, out_features=num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.inc_A1(x)
        x = self.inc_A2(x)

        x = self.inc_C1(x)
        x = self.inc_C2(x)

        x = self.layer2(x)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
