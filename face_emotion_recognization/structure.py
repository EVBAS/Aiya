import torch
from torch import nn


class cnn(nn.Module):
    def __init__(self, class_num, dropout_prob=0.5):
        super(cnn, self).__init__()
        # 输入图像大小为48x48
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, class_num)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()