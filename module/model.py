# 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class Simple_CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(Simple_CNN, self).__init__()

        # feature Extraxtion
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Classification
        self.fc1 = nn.Linear(64 * 18 *18, 128)
        self.fc2 = nn.Linear(128, num_classes)

    
    # 모델 구조
    def forward(self, x):
        # feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # classification
        x = x.view(-1, 64 * 18 * 18)    # tensor의 모양 바꾸기 / Fully Connected로 넘길때 사용
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x