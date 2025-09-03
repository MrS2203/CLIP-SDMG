import torch
from torch import nn


class SEModule(nn.Module):
    def __init__(self, teacher_dims, reduction=4):
        super().__init__()
        # 原始的视觉生成模块
        self.conv1 = nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1)

        # 通道注意力部分
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(teacher_dims, teacher_dims // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_dims // reduction, teacher_dims, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 可学习残差权重
        self.weights = nn.Parameter(torch.tensor(0.1, dtype=torch.float))

    def forward(self, x):
        b, c, _, _ = x.size()

        # 主干卷积 + ReLU
        x1 = self.conv1(x)
        y = self.avg_pool(x1)  # squeeze spatial
        y = self.fc(y)  # expand spatial
        x1 = x1 * y  # 通道注意力
        x1 = self.relu(x1)
        x1 = self.conv2(x1)  # 第二次卷积

        # 残差加权融合
        out = self.weights * x + x1
        return out
