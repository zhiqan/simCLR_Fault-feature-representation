import torch
import torch.nn as nn
import torch.nn.functional as F

class MimgnetEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(MimgnetEncoder, self).__init__()
        self.last_hidden_size = 2*2*hidden_size

        self.encoder = nn.Sequential(
            # -1, hidden_size, 42, 42
            nn.Conv2d(3, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 21, 21
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 10, 10
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, 2*hidden_size, 5, 5
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 2, 2
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.last_hidden_size)
        return h


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool_factor=1.0):
        super().__init__()
        stride = (int(2 * max_pool_factor))
        self.max_pool = nn.MaxPool1d(kernel_size=stride, stride=stride, ceil_mode=False)
        self.normalize = nn.BatchNorm1d(out_channels, affine=True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class ConvBase(nn.Sequential):
    def __init__(self, hidden=64, channels=1, layers=5, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, 3, max_pool_factor)]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, 3, max_pool_factor))
        super(ConvBase, self).__init__(*core)
        
        

class CNN4Backbone(ConvBase):
    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class NetCNN(torch.nn.Module):
    def __init__(self, output_size, hidden_size, layers, channels, embedding_size):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers, max_pool_factor=8// layers)
        self.classifier = torch.nn.Linear(embedding_size, output_size, bias=True)
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class CelebrEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(CelebrEncoder, self).__init__()
        self.last_hidden_size = 2*2*hidden_size

        self.encoder = nn.Sequential(
            # -1, hidden_size, 42, 42
            nn.Conv2d(3, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 21, 21
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 10, 10
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, 2*hidden_size, 5, 5
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
            # -1, hidden_size, 2, 2
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=False),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.last_hidden_size)
        return h