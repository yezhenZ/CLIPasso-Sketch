import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=16, output_dim=128):
        super(SimpleCNN, self).__init__()

        # 定义卷积层部分
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3),  # 输入：1通道，输出：16通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32

            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),  # 输入：16通道，输出：32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),  # 输入：32通道，输出：64通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # 展平操作，将卷积层输出展平成向量
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 16 * output_dim),

        )
        # 输出坐标点
        self.fc2 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 16 * 8),

        )

    def forward(self, x):
        # 通过卷积层部分
        x = self.features(x)
        # 展平输出
        x = self.flatten(x)
        # print(x.shape)
        # 通过全连接层
        x1 = self.fc(x)
        x2 = self.fc2(x)

        return x1, x2
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 64)
        self.gcn2 = GraphConvolution(64, 32)
        self.linear = nn.Linear(32, output_dim*2)
    def forward(self, X, adj):
        X = self.gcn1(X, adj)
        X = F.relu(X)  # Graph convolution + ReLU
        X = self.gcn2(X, adj)  # Output layer
        X= self.linear(X)
        X = nn.Sigmoid()(X)
        return X


class ConvNet(nn.Module):
    def __init__(self, specs, input_channels, deconv=False, keep_prob=1.0):
        """
        PyTorch版的卷积网络，支持卷积和反卷积。
        Args:
            specs: [(actfun, kernel_size, stride, out_channels), ...] 的配置列表。
            input_channels: 输入的通道数。
            deconv: 是否使用反卷积（默认为 False）。
            keep_prob: Dropout 保留概率（默认为 1.0）。
        """
        super(ConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.keep_prob = keep_prob
        in_channels = input_channels

        for i, (actfun, kernel_size, stride, out_channels) in enumerate(specs):
            if not deconv:
                self.layers.append(self.build_conv_layer(in_channels, out_channels, kernel_size, stride, actfun))
            else:
                self.layers.append(self.build_deconv_layer(in_channels, out_channels, kernel_size, stride, actfun))
            in_channels = out_channels

        if keep_prob < 1.0:
            self.dropout = nn.Dropout(1 - keep_prob)
        else:
            self.dropout = None

    def build_conv_layer(self, in_channels, out_channels, kernel_size, stride, actfun):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            self.select_act_func(actfun)
        ]
        return nn.Sequential(*layers)

    def build_deconv_layer(self, in_channels, out_channels, kernel_size, stride, actfun):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, output_padding=stride - 1),
            self.select_act_func(actfun)
        ]
        return nn.Sequential(*layers)

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return nn.Tanh()
        elif actfun == 'sigmoid':
            return nn.Sigmoid()
        elif actfun == 'relu':
            return nn.ReLU()
        else:
            return nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.keep_prob < 1.0 and self.dropout:
                x = self.dropout(x)
        return x


class FcNet(nn.Module):
    def __init__(self, specs, input_dim):
        """
        PyTorch版的全连接网络。
        Args:
            specs: [(actfun, in_features, out_features), ...] 的配置列表。
            input_dim: 输入的特征维度。
        """
        super(FcNet, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim

        for actfun, out_dim in specs:
            self.layers.append(self.build_fc_layer(in_dim, out_dim, actfun))
            in_dim = out_dim

    def build_fc_layer(self, in_dim, out_dim, actfun):
        layers = [
            nn.Linear(in_dim, out_dim),
            self.select_act_func(actfun)
        ]
        return nn.Sequential(*layers)

    def select_act_func(self, actfun):
        if actfun == 'tanh':
            return nn.Tanh()
        elif actfun == 'sigmoid':
            return nn.Sigmoid()
        elif actfun == 'relu':
            return nn.ReLU()
        else:
            return nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



